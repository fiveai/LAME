from collections import defaultdict
from collections.abc import Mapping
import numpy as np
import src.utils.comm as comm
from src.utils.events import EventWriter, get_event_storage
from src.utils.figures import adjust_xticks
from src.data import (
    DatasetCatalog
)
import matplotlib.pyplot as plt
import torch
__all__ = [
    "EvalHook",
    "DataSummarizer",
]


"""
Implement some common hooks.
"""


class HookBase:
    """
    Base class for hooks that can be registered with :class:`TrainerBase`.

    Each hook can implement 4 methods. The way they are called is demonstrated
    in the following snippet:
    ::
        hook.before_train()
        for iter in range(start_iter, max_iter):
            hook.before_step()
            trainer.run_step()
            hook.after_step()
        iter += 1
        hook.after_train()

    Notes:
        1. In the hook method, users can access ``self.trainer`` to access more
           properties about the context (e.g., model, current iteration, or config
           if using :class:`DefaultTrainer`).

        2. A hook that does something in :meth:`before_step` can often be
           implemented equivalently in :meth:`after_step`.
           If the hook takes non-trivial time, it is strongly recommended to
           implement the hook in :meth:`after_step` instead of :meth:`before_step`.
           The convention is that :meth:`before_step` should only take negligible time.

           Following this convention will allow hooks that do care about the difference
           between :meth:`before_step` and :meth:`after_step` (e.g., timer) to
           function properly.

    """

    """
    A weak reference to the trainer object. Set by the trainer when the hook is registered.
    """
    def before_episode(self):
        pass

    def after_episode(self):
        pass

    def before_train(self):
        """
        Called before the first iteration.
        """
        pass

    def after_train(self):
        """
        Called after the last iteration.
        """
        pass

    def before_step(self):
        """
        Called before each iteration.
        """
        pass

    def after_step(self, **kwargs):
        """
        Called after each iteration.
        """
        pass

    def state_dict(self):
        """
        Hooks are stateless by default, but can be made checkpointable by
        implementing `state_dict` and `load_state_dict`.
        """
        return {}


class DataSummarizer(HookBase):
    def __init__(self, datasets_dict, data_name, hook_name):
        self.dict = datasets_dict
        self.name = hook_name
        self.class_names = DatasetCatalog.get(data_name).thing_classes
        self.data_name = data_name

    def before_train(self):
        
        storage = get_event_storage()
        # obj_per_image = defaultdict(int)
        classes_histo = defaultdict(int)
        
        size_histo = defaultdict(int)
        for record in self.dict:
            # obj_per_image[len(record["annotations"])] += 1
            for obj in record["annotations"]:
                classes_histo[obj["supercategory_name"]] += 1
            if hasattr(record, "crop"):
                box = record["crop"]
                xmin, ymin, xmax, ymax = box
                h = ymax - ymin
                w = xmax - xmin
            else:
                h = record["height"]
                w = record["width"]
            str_size = f"{w}x{h}"
            size_histo[str_size] += 1 

        figures = []

        # ------------ Class distribution -----------------
        max_size = 30
        fig = plt.figure()
        values = [classes_histo[x] for x in self.class_names]
        total_samples = sum(values)

        sorted_indexes = np.argsort(values)[::-1]
        values = [values[i] for i in sorted_indexes][:max_size]
        named_xticks = [self.class_names[i] for i in sorted_indexes][:max_size]
        plt.bar(named_xticks, values)
        adjust_xticks(len(named_xticks))

        plt.xlabel("Classes")
        plt.ylabel("Count")
        plt.yscale('log')
        plt.title(f"{total_samples} samples in total.")
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        figures.append({"tag": f"{self.name} class distribution",
                        "figure": fig})
        if 'train' in self.name.lower():
            DatasetCatalog.get(self.data_name)._class_histo = values

        # ------------ Size distribution -----------------
        max_displayed = 20
        named_xticks = list(size_histo.keys())
        values = list(size_histo.values())
        num_sizes = min(max_displayed, len(named_xticks))
        fig = plt.figure()
        plt.bar(named_xticks[:max_displayed], values[:max_displayed])
        # fig.tight_layout()
        adjust_xticks(num_sizes)

        plt.xlabel("Classes")
        plt.ylabel("Count")
        plt.title(f"{len(named_xticks)} different resolutions in total.")
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        figures.append({"tag": f"{self.name} resolutions",
                        "figure": fig})

        # ------------ Put figures to storage -----------------
        storage.put_figures(figures)

    def before_adaptation(self):
        self.before_train()


class EpisodicWriter(HookBase):
    """
    Write events to EventStorage (by calling ``writer.write()``) periodically.

    It is executed every ``period`` iterations and after the last iteration.
    Note that ``period`` does not affect how data is smoothed by each writer.
    """

    def __init__(self, writers):
        """
        Args:
            writers (list[EventWriter]): a list of EventWriter objects
            period (int):
        """
        self._writers = writers
        for w in writers:
            assert isinstance(w, EventWriter), w

    def after_episode(self):
        for writer in self._writers:
            writer.write()

    def after_adaptation(self):
        self.after_train()

    def after_train(self):
        for writer in self._writers:
            # If any new data is found (e.g. produced by other after_train),
            # write them before closing
            writer.write()
            writer.close()


class EvalHook(HookBase):
    """
    Run an evaluation function periodically, and at the end of training.

    It is executed every ``eval_period`` iterations and after the last iteration.
    """

    def __init__(self, eval_period, eval_function):
        """
        Args:
            eval_period (int): the period to run `eval_function`. Set to 0 to
                not evaluate periodically (but still after the last iteration).
            eval_function (callable): a function which takes no arguments, and
                returns a nested dict of evaluation metrics.

        Note:
            This hook must be enabled in all or none workers.
            If you would like only certain workers to perform evaluation,
            give other workers a no-op function (`eval_function=lambda: None`).
        """
        self._period = eval_period
        self._func = eval_function

    def flatten_results_dict(self, results):
        """
        Expand a hierarchical dict of scalars into a flat dict of scalars.
        If results[k1][k2][k3] = v, the returned dict will have the entry
        {"k1/k2/k3": v}.

        Args:
            results (dict):
        """
        r = {}
        for k, v in results.items():
            if isinstance(v, Mapping):
                v = self.flatten_results_dict(v)
                for kk, vv in v.items():
                    r[k + "/" + kk] = vv
            else:
                r[k] = v
        return r

    def _do_eval(self, storage=None):
        if storage is None:
            storage = get_event_storage()
        results = self._func()

        # -------- Scalars -----------
        # ----------------------------
        key = "scalars"
        if results[key]:
            assert isinstance(
                results, dict
            ), "Eval function must return a dict. Got {} instead.".format(results[key])

            flattened_results = self.flatten_results_dict(results[key])
            for k, v in flattened_results.items():
                try:
                    v = float(v)
                except Exception as e:
                    raise ValueError(
                        "[EvalHook] eval_function should return a nested dict of float. "
                        "Got '{}: {}' instead.".format(k, v)
                    ) from e
            storage.put_scalars(**flattened_results, smoothing_hint=False)

        # -------- Arrays -----------
        # ----------------------------
        key = "arrays"
        if results[key]:
            assert isinstance(
                results, dict
            ), "Eval function must return a dict. Got {} instead.".format(results[key])

            for k, v in results[key].items():
                storage.put_array(k, v)

        # -------- Figures -----------
        # ----------------------------
        # key = "figures"
        # if results[key]:
        #     storage.put_figures(results[key])

        # Evaluation may take different time among workers.
        # A barrier make them start the next iteration together.
        comm.synchronize()

    def after_step(self, **kwargs):
        next_iter = self.iter + 1
        if self._period > 0 and self.iter % self._period == 0:
            # do the last eval in after_train
            if next_iter != self.max_iter:
                self._do_eval()

    def after_train(self):
        # This condition is to prevent the eval from running after a failed training
        # if self.iter + 1 >= self.max_iter:
        self._do_eval()
        # func is likely a closure that holds reference to the trainer
        # therefore we clean it to avoid circular reference in the end
        del self._func


class MetricHook(HookBase):
    """
    """
    def __init__(self, cfg):
        self.num_classes = len(DatasetCatalog.get(cfg.DATASETS.ADAPTATION[0]).thing_classes)
        self.reset()

    def reset(self):

        self.conf_matrix = torch.zeros(self.num_classes, self.num_classes).to(comm.get_rank())
        self.scalar_dic = defaultdict(list)
        self.array_dic = defaultdict(list)

    def before_episode(self):
        self.reset()

    def after_step(self, batched_inputs, outputs) -> None:
        """
        """
        preds = [instance.pred_classes[0] for instance in outputs["instances"]]
        gts = outputs["gts"]
        probas = outputs["probas"]
        # print(gts, preds, outputs["cls_head_logits"].argmax(-1))
        assert len(preds) == len(gts), (preds, gts)
        assert len(probas.size()) == 2 or len(probas.size()) == 3, probas.size()  # For ensembles

        self.array_dic["probs"].append(probas.detach().cpu())
        self.array_dic["gts"].append(gts.detach().cpu())
        for pred, gt in zip(preds, gts):
            self.conf_matrix[pred][gt] += 1
        self.scalar_dic["accuracy"].append((torch.diagonal(self.conf_matrix).sum() / self.conf_matrix.sum()).item())
        self.scalar_dic["nb_classes"].append(torch.unique(outputs['gts']).size(0))
        self.scalar_dic["label_entropy"].append(self.ent(outputs['probas'], 0).item())
        self.scalar_dic["cond_ent"].append(self.ent(outputs['probas']).item())
        self.scalar_dic["oracle_kl"].append(self.kl(outputs['one_hot_gts'], outputs['probas']).item())

    def after_episode(self, storage=None):
        if storage is None:
            storage = get_event_storage()

        self.array_dic["gts"] = torch.cat(self.array_dic["gts"]).numpy()  # [?,]
        self.array_dic["probs"] = torch.cat(self.array_dic["probs"], dim=0).numpy()  # [?, K]
        self.array_dic["conf_matrix"] = self.conf_matrix.cpu().numpy()  # [1, K, K]

        for k, v in self.array_dic.items():
            storage.put_array(k, v)

        for metric_name, array in self.scalar_dic.items():
            storage.put_array(f'{metric_name}', np.array(array))

    def ent(self, p, reduce_axis=None):

        if reduce_axis is not None:
            p = p.mean(reduce_axis)
        return -(p * torch.log(p + 1e-10)).sum(-1).mean()

    def kl(self, p1, p2, reduce_axis=None):
        assert p1.size() == p2.size()

        if reduce_axis is not None:
            p1 = p1.mean(reduce_axis)
            p2 = p2.mean(reduce_axis)

        return (p1 * torch.log(p1 / (p2 + 1e-10) + 1e-10)).sum()