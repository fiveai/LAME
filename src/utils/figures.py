import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
from types import SimpleNamespace
from src.data.catalog import DatasetCatalog
from typing import Dict
import yaml
from os.path import exists

plt.rc('font', **{'family': 'serif', 'serif': ['Times']})


persistent_dict = defaultdict(lambda: defaultdict(list))


class my_default_dict(dict):
    def __init__(self, fn):
        self.fn = fn

    def __missing__(self, key):
        return self.fn(key)


def fill_stacked_bars_plot(ax, method, model, labels, values, colors):

    assert len(labels) == len(values)
    bottom = 0
    print(method, model)
    model_delta_x = {'RN-18': -0.3, 'RN-50': -0.15, 'EN-B4': 0., 'RN-101': 0.15, 'ViT-B': 0.3}
    if method not in persistent_dict[ax]['method']:
        persistent_dict[ax]['method'].append(method)
    center = persistent_dict[ax]['method'].index(method)
    pos = center + model_delta_x[model]
    # print(method, model_delta_x, pos)

    for i, (label, value) in enumerate(zip(labels, values)):
        ax.bar([pos], [value], color=colors[i], edgecolor='white', width=0.1, bottom=[bottom], label=label)
        bottom += value

    ax.text(pos - 0.065, bottom + 0.01, model, size=20)
    n_methods_done = len(persistent_dict[ax]['method'])
    ax.set_xticks(np.arange(n_methods_done))
    ax.set_xticklabels(persistent_dict[ax]['method'])
    ax.set_ylabel('Runtime / Batch (s)')

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())


def consistency(matrix):
    n_classes = matrix.shape[0]
    diag_sum = np.diag(matrix).sum()
    consistency = (1 / n_classes) * diag_sum - (1 / ((n_classes - 1) * n_classes)) * (matrix.sum() - diag_sum)
    return consistency

    
def adjust_xticks(N: int):
    """
    N : number of xticks
    """
    plt.gca().margins(x=0)
    plt.gcf().canvas.draw()
    tl = plt.gca().get_xticklabels()
    maxsize = max([t.get_window_extent().width for t in tl])
    m = 0.2  # inch margin
    s = maxsize / plt.gcf().dpi * N + 2 * m
    margin = m / plt.gcf().get_size_inches()[0]

    plt.gcf().subplots_adjust(left=margin, right=1. - margin)
    plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])


def box_plot(ax, pos, data, edge_color, fill_color, **kwargs):

    bp = ax.boxplot(x=data, positions=pos, patch_artist=True, showmeans=False, 
                    meanprops={"marker": "s", "markersize": 3, "markerfacecolor": "white", "markeredgecolor": edge_color},
                    medianprops={'linewidth': 1},
                    boxprops={'linewidth': 1},
                    whiskerprops={'linewidth': 1}, **kwargs)
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)

    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)
    return bp
        

def fill_box_plot(ax, keys, values, label, count, N, color):
    spread = 0.3
    width = (2 * spread) / max(1, (N - 1))
    i = 0
    for i, array in enumerate(values):
        # if not isinstance(array, np.ndarray):
        #     array = np.array(array)
        all_ticks = i + np.linspace(-spread - 0.2, spread + 0.2, N)
        # if 'shortcut' not in layer:
        ticks = [all_ticks[count]]
        bp = box_plot(ax, ticks, array, 'black', color, widths=(width), showfliers=True, whis=1000)
    
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(keys)

    # Fetch current legend and add the method we just added
    if label is not None:
        persistent_dict[ax]['handles'].append(bp["boxes"][0])
        persistent_dict[ax]['labels'].append(label)
    return persistent_dict


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def violin_plot(ax, pos, data, color, w):
    data = np.sort(data)
    quartile1, _, quartile3 = np.percentile(data, [25, 50, 75])
    whiskers = np.array(adjacent_values(data, quartile1, quartile3))
    means = np.mean(data)
    whiskers_min, whiskers_max = whiskers[0], whiskers[1]

    vp = ax.violinplot(data, positions=pos, showmeans=False, showmedians=False,
                       showextrema=False, widths=(w), bw_method=0.3)
    # vp = sns.violinplot(data, positions=pos, showmeans=False, showmedians=False, showextrema=False, widths=(w), w = 0.2,  cut=2)
    ax.scatter(pos, means, marker='o', color='white', s=50, zorder=3)
    ax.vlines(pos, quartile1, quartile3, color='k', linestyle='-', lw=10)
    ax.vlines(pos, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

    for pc in vp['bodies']:
        pc.set_facecolor(color)
        pc.set_edgecolor('black')
        pc.set_alpha(0.6)

    return vp


def fill_violin_plot(ax, keys, values, label, count, N, color):
    spread = 0.2
    width = 0.8 * ((2 * spread) / max(1, (N - 1)))
    i = 0
    for i, array in enumerate(values):
        # if not isinstance(array, np.ndarray):
        #     array = np.array(array)
        all_ticks = i + np.linspace(-spread, spread, N)
        # if 'shortcut' not in layer:
        ticks = [all_ticks[count]]
        # bp = box_plot(ax, ticks, array, colors[count], 'white', widths=(width), showfliers=True, whis=1000)
        vp = violin_plot(ax, ticks, array, color=color, w=width)
    
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(keys)
    # ax.set_yscale('log')

    persistent_dict[ax]['handles'].append(vp["bodies"][0])
    persistent_dict[ax]['labels'].append(label)
    # ax.legend(persistent_dict[ax]['handles'], persistent_dict[ax]['labels'])

    return persistent_dict


def pad_arrays(array_list, pad_value=255, axis=0):
    array_list = [np.expand_dims(x, axis) for x in array_list]
    all_shapes = np.array([list(x.shape) for x in array_list])
    max_shape = all_shapes.max(axis=axis)
    for i, array in enumerate(array_list):
        for j, target_size in enumerate(max_shape):    
            if array.shape[j] != target_size:
                shape = array.shape
                pad_shape = list(shape)
                pad_shape[j] = target_size - shape[j]
                pad_array = pad_value * np.ones(pad_shape)
                array = np.concatenate([array, pad_array], axis=j)
            # array = self.pad_array(array=array, target_size=target_size, axis=j)
        array_list[i] = array
    return np.concatenate(array_list, axis=axis)


def is_option_relevant(args, method, option):
    config_root = Path('configs')
    all_method_configs = list(config_root.glob('**/*.yaml'))
    for cfg_path in all_method_configs:
        method_cfg = load_exp_config(config_path=cfg_path)
        if method_cfg.getattr('ADAPTATION.METHOD') == method:  # we have the current config here
            if method_cfg.hasattr(option):  # Then this option is actually relevant
                return True
            else:
                # print(f"Option {option} not relevant for method {method}")
                return False
    raise ValueError(f"Method {method} was not found in any of the following config files: {all_method_configs}")


def label_from_conf(args, cfg: SimpleNamespace):
    label = []
    key2pretty = my_default_dict(lambda key: key)
    key2pretty['ADAPTATION.LR'] = r'$\alpha$'
    key2pretty['ADAPTATION.METHOD'] = 'Method'
    for option in args.labels:
        if is_option_relevant(args, cfg.ADAPTATION.METHOD, option) or args.force_labels:
            value = cfg.getattr(option)
            # key = option.split('.')
            label.append(f"{key2pretty[option]}={value}")

    return ' / '.join(label)


def load_exp_config(folder: Path = None, config_path: Path = None):
    if config_path is None:
        assert folder is not None
        config_path = folder / 'config_0.yaml' if exists(folder / 'config_0.yaml') else folder / 'config.yaml'
    with open(config_path) as f:
        config_dic = yaml.load(f, Loader=yaml.FullLoader)

    # config_namespace = toNamespace(config_dic)
    config_namespace = Namespace(config_dic)
    return config_namespace


def get_classes(cfg):
    dataset = DatasetCatalog.get(cfg.DATASETS.ADAPTATION[0])
    if not dataset.is_mapping_trivial:
        dataset.filtered_unmapped_classes(cfg)
    classes = dataset.thing_classes
    return classes


def smooth(args, smoothing_alpha, metric, values: np.ndarray):
    alpha = smoothing_alpha[metric]
    EMA = np.zeros_like(values)
    EMA[0] = values[0]
    for i, x in enumerate(values[1:]):
        EMA[i + 1] = alpha * EMA[i] + (1 - alpha) * values[i + 1]
    return EMA


class Namespace(SimpleNamespace):
    def __init__(self, d: Dict):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, Namespace(v))
            else:
                setattr(self, k, v)
    def rec_setattr(self, k, v):
        key_list = k.split('.')
        d = self
        for key in key_list[:-1]:
            if not hasattr(d, key):
                setattr(d, key, Namespace({}))
            d = getattr(d, key)
        setattr(d, key_list[-1], v)
    def getattr(self, k):
        if self.hasattr(k):
            return eval(f"self.{k}")
        else:
            return 'N/A'
    def hasattr(self, k):
        try:
            eval(f"self.{k}")
            return True
        except:
            return False
    # Adding following functions to make it hashable
    def __eq__(self, other):
        if isinstance(self, SimpleNamespace) and isinstance(other, SimpleNamespace):
            return self.__dict__ == other.__dict__
        return NotImplemented
    def __hash__(self):
        return hash(str(self))
    def todict(self):
        d = vars(self)
        for k, v in d.items():
            if isinstance(v, Namespace):
                d[k] = v.todict()
        return d



# ============== RADAR FACTORY ==================


import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from matplotlib.path import Path as plt_path


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=plt_path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta