import os
import logging
import torch
from typing import Dict, List, Any, Optional
from tqdm import tqdm
from os.path import join

from src.engine import default_setup
from src.engine import hooks, try_get_key
from src.modeling import build_model
from src.utils import comm
from src.checkpoint import DetectionCheckpointer
from src.data import (
    build_datasets,
    build_test_loader)
from src.data.catalog import DatasetCatalog
from src.utils.logger import setup_logger
from src.utils.events import (
    EventStorage,
    JSONWriter,
    NumpyWriter,
    get_event_storage,
    TensorboardXWriter
)


class DefaultAdapter:
    """
    The base class from which all test-time adaptation methods inherit.
    """

    def __init__(self, cfg, args, **kwargs):
        """
        Args:
            cfg (CfgNode):
        """
        cfg.defrost()
        default_setup(cfg, args)

        # ---------- Setup loggers ----------

        rank = comm.get_rank()
        output_dir = try_get_key(cfg, "OUTPUT_DIR", "output_dir", "train.output_dir")
        if "setup_logger" in kwargs and kwargs["setup_logger"]:
            setup_logger(output_dir, distributed_rank=rank, name="fvcore")
            setup_logger(output_dir, distributed_rank=rank, name="src")
            self.logger = setup_logger(distributed_rank=rank, name=__name__)
        else:
            self.logger = logging.getLogger(__name__)

        # ---------- Build all datasets ----------

        self.all_datasets = build_datasets(cfg, cfg.DATASETS.ADAPTATION)

        # ---------- Define useful attributes ----------

        self._hooks: List[hooks.HookBase] = []
        self.start_iter = 0
        self.num_classes = len(DatasetCatalog.get(cfg.DATASETS.ADAPTATION[0]).thing_classes)
        self.rank = rank
        self.cfg = cfg

        # ---------- Build model and optimization stuff ----------

        self.model = build_model(cfg)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.checkpointer = DetectionCheckpointer(
                    # Assume you want to save checkpoints together with logs/statistics
                    model=self.model,
                    save_dir=self.output_dir)
        self.resume_or_load()

        # --------- Build hooks -------------

        self.register_hooks(self.build_hooks(self.all_datasets[0]))

    def run_full_adaptation(self):
        """
        """
        logger = logging.getLogger(__name__)
        logger.info("Running adaptation...")

        with EventStorage(0) as self.storage:

            self.before_adaptation()

            for run, dataset in enumerate(tqdm(self.all_datasets)):

                logger.info(f"Run {run + 1} / {len(self.all_datasets)}")
                self.num_samples = len(dataset)
                loader = build_test_loader(cfg=self.cfg,
                                           dataset_dict=dataset,
                                           batch_size=self.cfg.ADAPTATION.BATCH_SIZE,
                                           num_workers=self.cfg.DATALOADER.NUM_WORKERS,
                                           serialize=True,
                                           shuffle=False)
                self.before_episode()
                current_run_storage = self.run_episode(loader)
                self.after_episode(current_run_storage)
            self.after_adaptation()

    def run_episode(self, loader: torch.utils.data.DataLoader):
        """
        An 'episode' represents an independent run over a stream of data yielded by the loader.
        """
        raise NotImplementedError

    def resume_or_load(self):
        """
        The method will load model weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states).
        """
        path = self.cfg.MODEL.WEIGHTS
        _, incompatible = self.checkpointer.load(path, checkpointables=[])
        self.logger.info(incompatible)

    def build_hooks(self, dataset_dict):
        """
        Build a list of hooks to write numpy

        Returns:
            list[HookBase]:
        """
        ret = []
        data_name = self.cfg.DATASETS.ADAPTATION[0]
        self.metric_hook = hooks.MetricHook(self.cfg)
        ret.append(self.metric_hook)
        ret.append(hooks.DataSummarizer(dataset_dict, data_name, "Adaptation"))
    
        writers = [JSONWriter(join(self.output_dir, "metrics.json")),
                   TensorboardXWriter(log_dir=join(self.output_dir, 'tensorboard'),
                                      save_plots_too=self.cfg.SAVE_PLOTS,
                                      max_visu_frames=self.cfg.ADAPTATION.MAX_VISU_FRAMES),
                   ]
        if self.cfg.ADAPTATION.NUMPY_WRITER:
            writers.append(NumpyWriter(log_dir=join(self.cfg.OUTPUT_DIR, 'numpy')))
        ret.append(hooks.EpisodicWriter(writers))

        return ret

    def register_hooks(self, hooks: List[Optional[hooks.HookBase]]) -> None:
        """
        Register hooks to the adapter.

        Args:
            hooks (list[Optional[HookBase]]): list of hooks
        """
        hooks = [h for h in hooks if h is not None]
        self._hooks.extend(hooks)

    def before_adaptation(self):
        """
        Things to do before adaptation.
        """
        self.outer_iter = 0
        for h in self._hooks:
            fn = getattr(h, "before_adaptation", None)
            if callable(fn):
                h.before_adaptation()

    def before_episode(self):
        """
        Things to do before each episode.
        """
        self.inner_iter = 0
        for h in self._hooks:
            h.before_episode()

    def after_adaptation(self):
        """
        Things to do after adaptation.
        """

        for h in self._hooks:
            fn = getattr(h, "after_adaptation", None)
            if callable(fn):
                h.after_adaptation()

    def after_episode(self, run_storage):
        """
        Things to do after ever adaptation.
        """
        # --- Logging metrics ---
        for h in self._hooks:
            h.after_episode()
        self.storage.ingest_storage(run_storage)  # Ingest inner storage
        self.outer_iter += 1
        self.storage.iter = self.outer_iter

    def before_step(self):
        """
        Things to do before each step -- step=treatment of 1 batch--.
        """
        local_storage = get_event_storage()
        local_storage.iter = self.inner_iter

        for h in self._hooks:
            h.before_step()

    def after_step(self, batched_inputs: List[Dict[str, Any]], outputs: Dict[str, Any]):
        """
        Things to do after each step -- step=treatment of 1 batch--.
        """
        for h in self._hooks:
            h.after_step(batched_inputs=batched_inputs, outputs=outputs)
        self.inner_iter += 1

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`src.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model