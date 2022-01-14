from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt
from contextlib import contextmanager
import argparse
from typing import Tuple, Dict
import logging
import seaborn as sns
from shutil import copyfile
from .figures import load_exp_config, get_classes, smooth, \
    label_from_conf, pad_arrays, Namespace, \
    fill_stacked_bars_plot, my_default_dict
from .list_files import menu_selection
from os.path import join as ospjoin
from os.path import exists
from .calibration import expected_calibration_error


CB91_Blue = '#2CBDFE'
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'
colors = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber,
          CB91_Purple, CB91_Violet, 'r', 'm'] 

cmaps = [plt.get_cmap('Reds'), plt.get_cmap('Blues'), plt.get_cmap('Greens')]

method2pretty = my_default_dict(lambda method: method)
method2pretty['NonAdaptiveMethod'] = 'Baseline'
method2pretty['LAME'] = 'LAME'
method2pretty['Tent'] = 'TENT'
method2pretty['Shot'] = 'SHOT'
method2pretty['PseudoLabeller'] = 'PseudoLabel'

metric2pretty = my_default_dict(lambda metric: metric)
metric2pretty['accuracy'] = 'Accuracy'
metric2pretty['is_psd'] = 'Is kernel P.S.D ?'
metric2pretty['cond_ent'] = r'Entropy of predictions (nats)'

checkpoint2pretty = my_default_dict(lambda checkpoint: (checkpoint.split('.')[0].split('/')[-1]).upper())
checkpoint2pretty['checkpoints/vit/B-16.pth'] = 'ViT-B'
checkpoint2pretty['checkpoints/msra/R-18.pkl'] = 'RN-18'
checkpoint2pretty['checkpoints/msra/R-50.pkl'] = 'RN-50'
checkpoint2pretty['checkpoints/msra/R-101.pkl'] = 'RN-101'
checkpoint2pretty['checkpoints/pytorch/R-18.pth'] = 'RN-18'
checkpoint2pretty['checkpoints/pytorch/R-50.pth'] = 'RN-50'
checkpoint2pretty['checkpoints/pytorch/R-101.pth'] = 'RN-101'

styles = ['--', '-.', ':', '-', '--', '-.', ':', '-']
color_dic = my_default_dict(lambda k: colors[k % len(colors)])
style_dic = my_default_dict(lambda k: styles[k % len(styles)])
accumulate_dic: Dict[str, bool] = my_default_dict(lambda x: False if "mi_vs" in x else True)
accumulate_dic["conf_matrix"] = False
accumulate_dic["ECE"] = False
accumulate_dic["loss_deltas"] = False
accumulate_dic['channel_self_similarities'] = False
accumulate_dic['channel_cross_similarities'] = False

plot_type = defaultdict(lambda: 'standard')

smoothing_alpha: Dict[str, Dict[str, float]] = defaultdict(lambda: 0.99)

smoothing_alpha["mean_accuracy"] = 0.
# smoothing_alpha["adaptation"]["nb_pseudo_labelled_0"] = 0.99


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot training metrics')
    parser.add_argument('--stage', type=str, help='Which stage (benchmark/validation)')
    parser.add_argument('--dataset', type=str, help='Which dataset.')
    parser.add_argument('--latex', default=False, type=boolean_string, help='Bool type')
    parser.add_argument('--folder')
    parser.add_argument('--force_labels', action='store_true')
    parser.add_argument('--fontsize', type=int, default=11)
    parser.add_argument('--max_columns', type=int, default=2)
    parser.add_argument('--labels', nargs='+', type=str)
    parser.add_argument('--dpi', type=int, default=200,
                        help='Dots per inch when saving the fig')
    parser.add_argument('--out_extension', type=str, default='pdf',
                        help='Pdf for optimal quality')
    args = parser.parse_args()
    return args


# ===================== UTILS ======================
# ==================================================


def compute_confidence_interval(data: np.ndarray, ignore_value=None, axis=0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    assert len(data)
    if ignore_value is None:
        valid = np.ones_like(data)
    else:
        valid = (data != ignore_value)
    m = np.sum(data * valid, axis=axis, keepdims=True) / valid.sum(axis=axis, keepdims=True)
    # np.mean(data, axis=axis)
    std = np.sqrt(((data - m)**2 * valid).sum(axis=axis) / valid.sum(axis=axis))
    # std = np.std(data, axis=axis)

    pm = 1.96 * (std / np.sqrt(valid.sum(axis=axis)))

    m = np.squeeze(m).astype(np.float64)
    pm = pm.astype(np.float64)

    return m, pm


@contextmanager
def get_figure(args, figure_dic, metric_count, metric):

    # try:
    if metric not in figure_dic:
        count = 0
        if accumulate_dic[metric]:
            if plot_type[metric] == '3d':
                fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10), subplot_kw=dict(projection="3d"))
            else:
                fig = plt.Figure(figsize=(13, 10))
            ax = fig.gca()
        else:
            total_plots = metric_count[metric]
            n_rows = total_plots // args.max_columns + bool(total_plots % args.max_columns)
            n_columns = min(total_plots, args.max_columns)
            if plot_type[metric] == '3d':
                fig, axes = plt.subplots(nrows=n_rows, ncols=n_columns, figsize=(18, 8 * n_rows), subplot_kw=dict(projection="3d"))
            else:
                fig, axes = plt.subplots(nrows=n_rows, ncols=n_columns, figsize=(18, 8 * n_rows), sharex=True, sharey=True)
            ax = fig.axes[0]
    else:
        fig, count = figure_dic[metric]
        ax = fig.gca() if accumulate_dic[metric] else fig.axes[count]
    yield ax, count
    figure_dic[metric] = (fig, count + 1)

# ===================== MAIN ======================
# =================================================


def main(args=None, selected_folders=None, save_path=None, **kwargs):

    if args is None:

        args = Namespace(kwargs)

    if selected_folders is None:

        # ------- Find all possible folders --------

        if args.dataset == 'all':
            pattern = f"output/{args.stage}/**/{args.folder}/*"
        else:
            pattern = f"output/{args.stage}/{args.dataset}/**/{args.folder}/*"

        # ------- Ask user to select folders --------
        p = Path('./')
        files = set(p.glob(pattern))
        candidate_folders = list({Path(ospjoin(*file.parts[:-2])) for file in files})
        candidate_folders = list(filter(lambda folder: load_exp_config(folder).mode != 'tune', candidate_folders))

        selected_folders = menu_selection(candidate_folders)

    if save_path is None:
        print("Give a name for this series of plots? \n")
        exp_name = str(input())
        save_path = Path('plots') / args.stage / args.dataset / exp_name
    else:
        save_path = Path(save_path) / 'plots'

    save_path.mkdir(parents=True, exist_ok=True)

    # ------- Do the plots --------
    
    usable_files = []
    for path in selected_folders:  # type: ignore[union-attr]
        training_files = set(path.glob(f'**/{args.folder}/*'))
        usable_files.extend(list(training_files))

    metric_count = Counter([x.stem for x in usable_files])
    if "probs" in metric_count:
        metric_count["ECE"] = metric_count["probs"]
    usable_files.sort()
    figure_dic = {}  # type: ignore[var-annotated]
    for filepath in usable_files:
        make_plot(args=args,
                  figure_dic=figure_dic,
                  filepath=filepath,
                  metric_count=metric_count)

    # ------- Save the plots --------

    for metric, (fig, count) in figure_dic.items():
        for ax in fig.axes[count:]: fig.delaxes(ax)  # remove unused plots
        for ax in fig.axes:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        if not (plot_type[metric] == '3d'):  # idk why but bug with 3d axes when calling tight_layout
            fig.tight_layout()
        fig.savefig(save_path / '{}.{}'.format(metric,
                                               args.out_extension),
                    dpi=args.dpi,
                    bbox_inches='tight')

    # ------- Also save config -------
    config_path = path / 'config_0.yaml' if exists(path / 'config_0.yaml') else path / 'config.yaml'
    copyfile(config_path, save_path / "config.yaml")


# ===================== ADAPTATION ======================
# =======================================================

def make_plot(args,
              figure_dic: Dict[str, plt.Figure],
              filepath: Path,
              metric_count: Dict[str, int]
              ) -> None:

    metric = filepath.stem
    config = load_exp_config(filepath.parent.parent)

    # ------------- Conf matrix -------------
    if metric == "conf_matrix":
        max_size = 30
        with get_figure(args, figure_dic, metric_count, metric) as (ax, count):
            conf_matrix = np.load(filepath).squeeze()[..., :max_size, :max_size]  # [T, K, K]
            if len(conf_matrix.shape) == 2:
                assert conf_matrix.shape[0] == conf_matrix.shape[1]
                conf_matrix = np.expand_dims(conf_matrix, axis=0)
            assert len(conf_matrix.shape) == 3, conf_matrix.shape
            norm_conf_matrix = conf_matrix / conf_matrix.sum()
            # classes = MetadataCatalog.get("imagenet_vid_2015_train").thing_classes
            classes = get_classes(config)[:max_size]
            assert len(classes) == conf_matrix.shape[1]
            sns.heatmap(norm_conf_matrix.sum(axis=0), ax=ax, cmap="afmhot", vmin=0, vmax=0.1)
            ax.set_xticks(np.arange(len(classes)) + 0.5)
            ax.set_yticks(np.arange(len(classes)) + 0.5)
            ax.set_xticklabels(classes, rotation=90)
            ax.set_yticklabels(classes, rotation=0)
            ax.set_xlabel("True class")
            ax.set_ylabel("Predicted class")
            ax.grid(False)
            y_pos = 1.5 if accumulate_dic[metric] else 0.95
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, y_pos))

    # ------------- Calibration metrics -------------------

    elif metric == "probs":
        metric = "ECE"
        with get_figure(args,figure_dic, metric_count, metric) as (ax, count):
            parents = Path(ospjoin(*filepath.parts[:-1]))
            gts = np.concatenate(np.load(parents / "gts.npy", allow_pickle=True))
            probas = np.concatenate(np.load(parents / "probs.npy", allow_pickle=True))

            bins, histo, density, ECE = expected_calibration_error(gts, probas)
            mid = (bins[1:] + bins[:-1]) / 2
            assert mid.shape == histo.shape, (mid.shape, histo.shape)
            reds = plt.get_cmap('Reds')  # this returns a colormap
            color_fade = [reds(x) for x in density]  # blues(x) returns a c
            ax.bar(mid, histo, width=0.05, edgecolor='black', label=f"ECE={ECE}", color=color_fade)
            ax.plot([0, 1], [0, 1], linestyle='--', color='r', linewidth=1.5, label='Perfect calibration')
            for i, (x, y) in enumerate(zip(mid, histo)):
                ax.text(x - 0.02, y + 0.02, np.round(density[i], 3), size=6)
            ax.set_xlabel("Confidence")
            ax.set_ylabel("Accuracy")
            ax.grid(False)
            y_pos = 1.3 if accumulate_dic[metric] else 0.95
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, y_pos), prop={"size": 20})

    elif metric == "eigenvalues":

        with get_figure(args, figure_dic, metric_count, metric) as (ax, count):
            eig = np.load(filepath, allow_pickle=True)  # [n_runs, n_batches, batch_size]
            eig = np.diagflat(eig.mean(axis=(0, 1)))

            sns.heatmap(eig, ax=ax, cmap="afmhot", cbar=True, annot=True)
            ax.grid(False)
            y_pos = 1.5 if accumulate_dic[metric] else 0.95
            ax.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False)  # labels along the bottom edge are off
            ax.tick_params(
                axis='y',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False)  # labels along the bottom edge are off

    # ------------- Time plots -------------

    elif metric == 'forward_time':
        try:
            forward = np.load(filepath)  # [T, N_ITER] or [2, T]
            optim = np.load(str(filepath).replace(metric, 'optimization_time'))
            inf = np.load(str(filepath).replace(metric, 'inference_time'))

            with get_figure(args, figure_dic, metric_count, 'times') as (ax, count):
                method = config.ADAPTATION.METHOD
                model = checkpoint2pretty[config.MODEL.WEIGHTS]
                labels = [r'$1^{st}$ Forward', 'Optimization', '$2^{nd}$ Forward']
                values = [array.mean() for array in [forward, optim, inf]]
                fill_stacked_bars_plot(ax, method2pretty[method], model, labels, values, color_dic)
                # ax.set_yscale('log')
        except:
            logger = logging.getLogger(__name__)
            logger.warning("Could not plot the times. Likely because time arrays are not squared (may happen if Benchmark mode active")

    # ------------- All other metrics -------------

    elif metric not in ["gts", "inference_time", "optimization_time"]:
        with get_figure(args, figure_dic, metric_count, metric) as (ax, count):
            try:
                array = np.load(filepath)  # [T, N_ITER] or [2, T]
            except:
                array_list = np.load(filepath, allow_pickle=True)  # [T, N_ITER] or [2, T]
                array = pad_arrays(array_list, axis=0)
            # assert len(array.shape) == 2, (filepath, array.shape)

            if array.shape[0] == 2:  # Corresponds to outer scalars
                x = array[0, :]
                y = array[1, :]
                smoothed_y = smooth(args, smoothing_alpha, metric, y)
                label = label_from_conf(args, config)
                if len(smoothed_y) == 1:
                    ax.scatter(x, smoothed_y, label=label, color=color_dic[count], linestyle=style_dic[count])
                else:
                    ax.plot(x, smoothed_y, label=label, color=color_dic[count], linestyle=style_dic[count])
                    ax.fill_between(x, smoothed_y, y, color=color_dic[count], alpha=0.2)

                ax.set_xlabel("Episodes")
            else:  # Corresponds to inners scalars
                mean, conf_inter = compute_confidence_interval(array, ignore_value=255, axis=0)  # [N_ITER]
                label = label_from_conf(args, config)
                timesteps = np.arange(array.shape[1])
                ax.plot(timesteps, mean, label=label, color=color_dic[count], linestyle=style_dic[count], linewidth=6)
                ax.fill_between(timesteps, mean - conf_inter, mean + conf_inter, color=color_dic[count], alpha=0.1)
                ax.set_xlabel("Online batches")
                ax.set_ylabel(metric2pretty[metric])
            # ax.grid(axis='y')
            y_pos = 1.2 if accumulate_dic[metric] else 0.95
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, y_pos), frameon=False, ncol=4)

    if not accumulate_dic[metric]:
        ax.set_title(label_from_conf(args, config))


if __name__ == "__main__":

    args = parse_args()

    if args.latex:
 
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"]})
        # for Palatino and other serif fonts use:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Palatino"],
        })

        plt.rcParams.update({'font.size': 35})
       
    main(args=args)
