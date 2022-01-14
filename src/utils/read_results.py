from pathlib import Path
from collections import defaultdict
import numpy as np
import argparse
from typing import Dict
import seaborn as sns
import os
from os.path import join as ospjoin
from .figures import load_exp_config, radar_factory, Namespace
import matplotlib.pyplot as plt
from src.utils.logger import setup_logger
import yaml
from matplotlib.patches import Rectangle
from .plot import compute_confidence_interval, boolean_string, checkpoint2pretty, method2pretty, my_default_dict
from .figures import fill_box_plot

colors = ['tab:blue', 'tab:brown', 'tab:green', 'tab:red', 'tab:purple', 'tab:pink']
method2color = my_default_dict(lambda method: colors[hash(method) % len(colors)])
method2color['NonAdaptiveMethod'] = colors[0]
method2color['Shot'] = colors[1],
method2color['PseudoLabeller'] = colors[2]
method2color['Tent'] = colors[3]
method2color['LAME'] = colors[4]
method2color['AdaBN'] = colors[5]

# method2color['LAME AFFINITY=kNN '] = colors[0]
# method2color['LAME AFFINITY=linear '] = colors[1]
# method2color['LAME AFFINITY=rbf '] = colors[2]

dataset2pretty = {'imagenet_val': 'ImageNet-Val', 'imagenet_c_test': 'ImageNet-C', 'imagenet_v2': 'ImageNet-V2',
                  'imagenet_c_val': 'ImageNet-C', 'imagenet_vid_val': 'ImageNet-Vid', 'tao_trainval': 'TAO-LaSOT',
                  'imagenet_c_16': r'ImageNet-C$_{16}$'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot training metrics')
    parser.add_argument('--stage', type=str, help='Which mode')
    parser.add_argument('--datasets', type=str, nargs='+')
    parser.add_argument('--cases', type=str, nargs='+')
    parser.add_argument('--case_names', type=str, nargs='+')
    parser.add_argument('--out_name', type=str)
    parser.add_argument('--action', type=str, required=True)
    parser.add_argument('--methods', type=str, nargs='+', default='all')
    parser.add_argument('--method_params', type=str, nargs='+', default='',
                        help='When you want to plot different versions of the same method.')
    parser.add_argument('--save_dir', type=str, default=ospjoin('output', 'validation'))
    parser.add_argument('--set_title', type=boolean_string, default="True")
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--save_name', type=str)
    parser.add_argument('--latex', default=False, type=boolean_string, help='Bool type')
    parser.add_argument('--extension', default='pdf')
    parser.add_argument('--title', type=str, default='')
    parser.add_argument('--metric', type=str, default='accuracy', help='')
    args = parser.parse_args()
    return args


# =================== Utilities  ===================
# ==================================================

def default_to_regular(d):
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d

    
def read_metric(metric: str, folder: Path, logger):
    """
    Accumulates results for each method
    """
    metrics_path = folder / 'numpy' / f'{args.metric}.npy'
    if metrics_path.exists():
        array = np.load(metrics_path, allow_pickle=True)
        array = np.array([x[-1] for x in array])
        return array
    else:
        logger.warning(f"Path {metrics_path} does not exist.")
        return None


def nested_dd(type, target_depth, cur_depth=1):
    if cur_depth == target_depth:
        return defaultdict(type)
    else:
        return defaultdict(lambda: nested_dd(type, target_depth, cur_depth + 1))


def conf2case(args, conf, folder, logger):
    """
    Filters out irrelevant folders, and only return the corresponding case if there is any.
    """
    for i, (case_specs, case_name) in enumerate(zip(args.cases, args.case_names)):

        case_args = [t.split('=') for t in case_specs.split(',')]  # ["MODEL.ARGS="]
        l = []
        for key, value in case_args:
            l.append(eval(f'conf.{key}') == eval(value))
        if all(l):
            return case_name
    # logger.warning(f"Folder {folder} does not match any specified case. Ignoring")
    return None


def conf2dataset(args, conf, logger):
    """
    Filters out irrelevant folders, and only return the corresponding case if there is any.
    """
    for i, (dataset) in enumerate(args.datasets):
        if dataset in eval('conf.DATASETS.ADAPTATION'):
            return dataset
    return None


def read_all_results(args: argparse.Namespace, logger):
    """
    Here, we first gather all relevant results (i.e that match the cases described in case_names).
    Returns:

    dict[case][method][conf]
    """
    logger.info("Reading all relevant results")
    p = Path('./')
    files = p.glob(f"output/{args.stage}/**/accuracy.npy")
    candidate_folders = list({Path(ospjoin(*file.parts[:-2])) for file in files})

    all_results = nested_dd(list, 4)
    for folder in candidate_folders:
        config = load_exp_config(folder=folder)
        corresponding_case = conf2case(args, config, folder, logger)
        if corresponding_case is not None:
            array = read_metric(args.metric, folder, logger)
            if array is not None:
                method = config.ADAPTATION.METHOD
                if args.methods == 'all' or method in args.methods:
                    keep_method = True
                else:
                    keep_method = False

                if keep_method:
                    if len(args.method_params):
                        assert args.methods != 'all' and len(args.methods) == 1
                        for param in args.method_params:
                            value = eval(f'config.{param}')
                            pretty_param = param.split('.')[-1].split('_')[-1]
                            method += f' {pretty_param}={value} '
                    all_results[corresponding_case][method][config]['all_values'] = array
                    m, pm = compute_confidence_interval(array)
                    all_results[corresponding_case][method][config]['mean'] = m
                    all_results[corresponding_case][method][config]['pm'] = pm

    logger.info("Finished reading all relevant results")
    return default_to_regular(all_results)


def get_per_case_best_config(args: argparse.Namespace, all_results: Dict, logger):

    logger.info("Obtained best configurations")

    # -------- Print the best configurations -----------
    adaptive_results = nested_dd(list, 4)
    baseline_results = {}

    for case in all_results:

        sorted_items = sorted(all_results[case].items())
        baseline_configs = list(all_results[case]['NonAdaptiveMethod'].keys())
        assert len(baseline_configs) == 1, "There should be a single config for each case."
        baseline_results[case] = all_results[case]['NonAdaptiveMethod'][baseline_configs[0]]['mean']

        for method, method_results in sorted_items:
            if method != 'NonAdaptiveMethod':
                sorted_confs = [k for k, _ in sorted(method_results.items(), key=lambda t: t[1]['mean'])]  # sorted by ascending metric value
                best_conf = sorted_confs[-1]
                worst_conf = sorted_confs[0]

                for type_, config in zip(['best', 'worst'], [best_conf, worst_conf]):

                    res = get_tuned_params(config)
                    adaptive_results[case][method][type_]['hparams'] = res['hparam_indexes']
                    adaptive_results[case][method][type_]['config'] = config
                    adaptive_results[case][method][type_]['mean'] = method_results[config]['mean']
                    adaptive_results[case][method][type_]['pm'] = method_results[config]['pm']

    logger.info("Done obtaining best configurations")
    return default_to_regular(baseline_results), default_to_regular(adaptive_results), config.ADAPTATION.HPARAMS2TUNE, config.ADAPTATION.HPARAMS_VALUES


def get_tuned_params(config):
    params2tune = config.ADAPTATION.HPARAMS2TUNE
    tried_values = config.ADAPTATION.HPARAMS_VALUES

    msg = []
    res = {}

    indexes = []
    hparam_config = Namespace({})
    for key, tried in zip(params2tune, tried_values):
        value = eval(f"config.{key}")
        hparam_config.rec_setattr(key, value)
        value_index = tried.index(value)
        current_msg = f"{key}={value}"
        if value_index == 0 or value_index == len(tried) - 1:
            current_msg += "[!]"  # To signify that the value is on the border of the current grid
        indexes.append(value_index)
        msg.append(current_msg)
    msg = '\t'.join(msg)
    res['msg'] = msg
    res['tot_configs'] = np.prod([len(grid) for grid in tried_values])
    res['hparam_indexes'] = indexes
    res['hparam_config'] = hparam_config
    return res

# ===================================== Validation plots =======================================
# ==============================================================================================


def hparams_spider_chart(args, baseline_results, adaptive_results, hparams2tune, hparams_value1s, logger):
    """
    Make spider chart of best/worst configuration. 
    """
    N = len(hparams2tune)
    n_values = np.array([len(x) for x in hparams_values])
    var_labels = [x.replace('ADAPTATION.', '') for x in hparams2tune]
    print(var_labels, n_values)
    theta = radar_factory(N, frame='polygon')

    n_cases = len(adaptive_results)
    methods = [method for method in list(adaptive_results.values())[0].keys() if method != 'MMLS']
    n_columns = len(methods)
    fig, axs = plt.subplots(figsize=(5 * n_columns, 5 * n_cases), nrows=n_cases, ncols=n_columns, subplot_kw=dict(projection='radar'),
                            squeeze=False, sharey=True, sharex=True)
    fig.subplots_adjust(wspace=0.15, hspace=0.1)

    for i, case in enumerate(sorted(adaptive_results)):
        dic = adaptive_results[case]

        colors = ['b', 'r', 'g', 'm', 'y']
        # Plot the four cases from the example data on separate axes
        for j, method in enumerate(methods):
            if i == 0:
                axs[i, j].set_title(method, weight='bold', size='large')
            if j == 0:
                axs[i, j].set_ylabel(case, fontsize=15)    
                axs[i, j].yaxis.set_label_coords(-0.1, 0.5)
            axs[i, j].set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0])
            for label, color in zip(dic[method], colors):
                hparam_indexes = dic[method][label]['hparams']
                metric_value = np.round(100 * (dic[method][label]['mean'] - baseline_results[case]['mean']), 2)
                norm_index = np.clip((np.array(hparam_indexes)) / (n_values - 1), 0.05, 1.0)
                print(method, case, norm_index)
                axs[i, j].plot(theta, norm_index, color=color, label=label + f' ({metric_value})')
                axs[i, j].fill(theta, norm_index, facecolor=color, alpha=0.25)
            axs[i, j].set_varlabels(var_labels)
            axs[i, j].legend(loc=(0.75, .8), labelspacing=0.1, fontsize=12)

    # add legend relative to top-left plot
    # labels = list(adaptive_results[method].keys())
    root = ospjoin('plots', args.stage)
    os.makedirs(root, exist_ok=True)
    plt.tight_layout()
    plt.savefig(ospjoin(root, f'{args.out_name}.{args.extension}'))


def make_cross_cases_heatmap(args, adaptive_results, baseline_results, all_results, logger):

    logger.info("Building the cross-cases heatmap ...")

    all_methods = list(list(adaptive_results.values())[0].keys())
    all_methods.sort(reverse=True)
    grouped_results = defaultdict(int)
    for case, case_name in zip(args.cases, args.case_names):
        data = case.split(',')[0].split('=')[-1]
        data = dataset2pretty[eval(data)[0]]
        grouped_results[data] += 1

    n_cases = len(args.cases)
    # index2case = {({chr(97 + i)}): case_name for i, case_name in enumerate(all_cases)}

    logger.info(f"Methods detected : {all_methods}")

    for k, method in enumerate(all_methods):
        cross_matrix = np.zeros((n_cases, n_cases))
        for i, tune_case in enumerate(args.case_names):
            if method in adaptive_results[tune_case]:
                best_config = adaptive_results[tune_case][method]['best']['config']
                for j, test_case in enumerate(args.case_names):

                    found_matching_config = False
                    if method in all_results[test_case]:
                        for other_config in all_results[test_case][method]:
                            # Check if the other_config matches the best config on tuned hyper-parameters
                            match = True
                            for param in best_config.ADAPTATION.HPARAMS2TUNE:
                                if eval(f'other_config.{param}') != eval(f'best_config.{param}'):
                                    match = False
                            if match:
                                result = all_results[test_case][method][other_config]['mean'] - baseline_results[test_case]
                                cross_matrix[i, j] = result
                                found_matching_config = True
                                break
                    else:
                        logger.warning(f"Could not find results for method {method} for test case {test_case}")
                    if not found_matching_config:
                        logger.warning(f"({i}, {j}) position of the matrix not filled.")
            else:
                logger.warning(f"Whole row {i} not filled for method {method}")

        cross_matrix = 100 * cross_matrix
        cross_matrix = np.round(cross_matrix, 1)
        logger.info(cross_matrix)

        # Preparing canvas
        fig = plt.figure(figsize=(11 + (k == len(all_methods) - 1), 12))
        ax = fig.gca()
        cmap = sns.diverging_palette(10, 133, as_cmap=True)

        # Doing the plot
        sns.heatmap(cross_matrix, ax=ax, cmap=cmap, vmin=-10, vmax=10,
                    annot=True, cbar=(k == len(all_methods) - 1), cbar_kws={'shrink': 0.8, 'pad': 0.03})

        # Taking care of ticks and axis naming
        tick_width = 2
        tick_length = 30
        letter_hpad = 0.5
        letter_vpad = 0.4
        data_hpad = 1.2
        data_vpad = 0.7
        ticks = np.cumsum([0] + [grouped_results[data] for data in grouped_results])
        ax.set_xticks(ticks[1:-1])
        for i in range(n_cases): ax.text(i + letter_vpad, n_cases + letter_hpad, chr(97 + (i % 4)).upper())
        for i, data in enumerate(grouped_results): ax.text(ticks[i] + data_vpad, n_cases + data_hpad, data)
        ax.tick_params(axis='x', which='both', labelbottom=False, width=tick_width, length=tick_length)
        if k == 0:
            ax.set_yticks(ticks[1:-1])
            for i in range(n_cases): ax.text(- letter_hpad, n_cases - (i + letter_vpad), chr(97 + ((-i - 1) % 4)).upper())
            for i, data in enumerate(list(grouped_results.keys())[::-1]): ax.text(- data_hpad, n_cases - (ticks[i] + data_vpad), data, rotation=90)
            ax.tick_params(axis='y', which='both', width=tick_width, length=tick_length)
        else:
            ax.tick_params(axis='y', which='both', left=False, right=False, labelright=False, labelleft=False)
        if args.set_title:
            ax.set_title(rf"({chr(97 + k)}) \textbf{{{method2pretty[method]}}}", y=1.05)
        ax.grid(False)

        # Highlight diagonal
        for i in range(n_cases):
            ax.add_patch(Rectangle((i, i), 1, 1, fill=False, edgecolor='black', lw=5))

        # Saving figure

        root = ospjoin('plots', args.stage)
        os.makedirs(root, exist_ok=True)
        path = ospjoin(root, f'{method}_cross_cases.{args.extension}')
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        logger.info(f"Save plot at {path}")
        plt.clf()


def get_per_method_best_config(all_results, logger):

    all_cases = list(all_results.keys())

    all_methods = list(set([method for case in all_cases for method in all_results[case].keys()]))

    logger.info(f"Methods detected : {all_methods}")

    method2bestconfig = {}

    for method in all_methods:
        available_cases = [case for case in all_cases if method in all_results[case]]
        config2metrics = defaultdict(list)

        # ---- Obtain all configs that at least appear once somewhere ----
        available_configs = list(set([config for case in available_cases for config in all_results[case][method].keys()]))

        # ---- Only keep configs for which all cases have already been done ----
        # finished_configs = [config for config in available_configs if all([config for case in all_cases in all_results[case][method]])]
        missing_exp = 0
        for config in available_configs:
            res = get_tuned_params(config)
            for case in available_cases:
                if config in all_results[case][method]:
                    config2metrics[res['hparam_config']].append(all_results[case][method][config]['mean'])
                else:
                    missing_exp += 1
        tot_configs = res['tot_configs']
        missing_exp += len(all_cases) * (tot_configs - len(available_configs))

        # ---- Now average the results over all cases ----
        # if method == 'AdaBN':
        #     import pprint
        #     pp = pprint.PrettyPrinter(indent=4)
        #     pp.pprint(config2metrics)
        for config in config2metrics:
            all_values = config2metrics[config]
            config2metrics[config] = np.mean(all_values)
        print(config2metrics)
        # ---- Get the best overall config ----
        accs = list(config2metrics.values())
        best_index = np.argmax(accs)
        (best_config, best_mean) = list(config2metrics.items())[best_index]
        n_configs = len(config2metrics)
        method2bestconfig[method] = {'metric_mean': best_mean,
                                     'n_configs': n_configs,
                                     'missing_exp': missing_exp,
                                     'total_exp': len(all_cases) * tot_configs,
                                     'best_config': best_config}

    return method2bestconfig


def log_best_metrics(method2bestconfig, logger):
    """
    For each method, log the number of experiments missing (according to the specified grid of parameters),
    as well as the current best config and best mean.
    """
    print("\n")
    for method, res in method2bestconfig.items():
        logger.info(f"============ Method {method} =========== \n")
        logger.info(f"Missing experiments {res['missing_exp']} (out of {res['total_exp']})")
        logger.info(f"{res['n_configs']} configs tried. Best mean is {np.round(100 * res['metric_mean'], 2)}")
        logger.info(f"{res['best_config']} \n")
    print("\n")


# ===================================== Benchmarking plots =====================================
# ==============================================================================================

def get_benchmark_dict(all_results, logger, groupby, checks=[]):
    """
    Essentially re-formats the results for exploitation in benchmark plots, and makes necesarry verifications.
    The resulting dic is in form benchmark_dic[method][groupby] = {'mean': x, 'values: [run1, run2, ...] '}

    args:
        all_results: The dictionnary of all results
        groupby: A key (e.g MODEL.DEPTH) used to group the different cases for each method
    """
        
    benchmark_dic = nested_dd(list, 3)
    groupby_values = []
    for case in all_results:
        for (method, config_dic) in all_results[case].items():

            for check in checks:
                check(case, method, config_dic)
            available_configs = list(config_dic.keys())
            all_values = [v for c in available_configs for v in config_dic[c]['all_values']]
            mean = np.mean([config_dic[c]['mean'] for c in available_configs])
            pm = np.mean([config_dic[c]['pm'] for c in available_configs])

            groupby_val = eval(f'available_configs[0].{groupby}')
            if isinstance(groupby_val, list):
                logger.info(f"Groupby key {groupby} is a list (non hashable), so taking the first element ({groupby_val[0]}) as key.")
                groupby_val = groupby_val[0]

            # if len(all_values) < 10:
            #     logger.warning(f"Case {case} for {method} only contains {len(all_values)} values")
            benchmark_dic[method][groupby_val]['values'] = all_values
            benchmark_dic[method][groupby_val]['mean'] = mean
            benchmark_dic[method][groupby_val]['pm'] = pm
            groupby_values.append(groupby_val)
    return benchmark_dic, list(set(groupby_values))


def benchmark_box(args, all_results, logger):
    """
    For each case and each method, plot the box corresponding to all different runs.
    """

    def check_1(case, method, config_dic):
        assert len(config_dic) == 1, f"Cases should be defined such that there is a single config per case. \
                                      Currently {len(config_dic)} configs for {method} / {case}"

    def check_2(case, method, config_dic):
        c = list(config_dic.keys())[0]
        if len(config_dic[c]['all_values']) < 10:
            logger.warning(f"Case {case} for {method} only contains {len(config_dic[c]['all_values'])} values")

    benchmark_dic, datasets = get_benchmark_dict(all_results, logger, 'DATASETS.ADAPTATION', checks=[check_1, check_2])

    methods = sorted(list(benchmark_dic.keys()))  # for consistency across plots
    datasets.sort()  # to make sure we always have them in the same order
    per_method_avg = [np.mean([benchmark_dic[m][d]['mean'] for d in datasets]) for m in methods]
    best_method_index = np.argmax(per_method_avg)

    fig, axs = plt.subplots(figsize=(12, 10), nrows=1, ncols=len(datasets))
    for j, d in enumerate(datasets):
        if len(datasets) > 1:
            ax = axs[j]
        else:
            ax = axs
        ax.grid(axis='y')
        new_min, new_max = 1., 0.
        for i, method in enumerate(methods):
            values = benchmark_dic[method][d]['values']
            new_min, new_max = min(min(values), new_min), max(max(values), new_max)
            cross_cases_mean = np.round(100 * per_method_avg[i], 1)
            label = f"{method2pretty[method]} ({cross_cases_mean})"
            label = rf"\textbf{{{label}}}" if (i == best_method_index) else label
            dic = fill_box_plot(ax,
                                keys=[dataset2pretty[d]],
                                values=[values],
                                label=label if (j == len(datasets) - 1) else None,
                                count=i,
                                N=len(methods),
                                color=method2color[method])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        ax.set_ylim(new_min - 0.01, new_max + 0.01)
    if args.set_title:
        fig.suptitle(rf"\textbf{{{args.title}}}", y=1.1, fontsize=30)
    fig.legend(dic[ax]['handles'], dic[ax]['labels'],
               loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3,
               prop=dict(size=23), frameon=False)
    fig.tight_layout()
    root = ospjoin('plots', 'benchmark')
    os.makedirs(root, exist_ok=True)
    fig.savefig(ospjoin(root, f'box_{args.save_name}.{args.extension}'), dpi=300, bbox_inches='tight')


def benchmark_batch(args, all_results, logger):
    """
    For each case and each method, plot the box corresponding to all different runs.
    """

    # def check_1(case, method, config_dic):
    #     assert len(config_dic) == 1, f"Cases should be defined such that there is a single config per case. \
    #                                   Currently {len(config_dic)} configs for {method} / {case}"

    def check_1(case, method, config_dic):
        for c in config_dic:
            assert len(config_dic[c]['all_values']) >= 1, f"Case {case} for {method} only contains {len(config_dic[c]['all_values'])} values \
                        for {c.DATASETS.ADAPTATION}"

    benchmark_dic, batch_sizes = get_benchmark_dict(all_results, logger, 'ADAPTATION.BATCH_SIZE', checks=[check_1])
    batch_sizes.sort(reverse=True)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.gca()

    for i, method in enumerate(benchmark_dic):
        means = np.array([benchmark_dic[method][b]['mean'] for b in batch_sizes])
        pm = np.array([benchmark_dic[method][b]['pm'] for b in batch_sizes])
        # print(values.shape, "-----------------------")
        x = np.arange(len(batch_sizes))
        if method == 'NonAdaptiveMethod':
            ax.hlines(means[0], x[0], x[-1], label=method2pretty[method], color=method2color[method],
                      linewidth=4, linestyle='--')
        else:
            ax.plot(x, means, label=method2pretty[method], c=method2color[method], marker='x', linewidth=1, markersize=10, markeredgewidth=3)
            ax.fill_between(x, means - pm, means + pm, color=method2color[method], alpha=0.1)
    ax.legend(
        loc='center',
        bbox_to_anchor=[0.5, 1.15],       # bottom-right
        ncol=3,
        frameon=False,     # don't put a frame
    )
    ax.set_xticks(np.arange(len(batch_sizes)))
    ax.set_xticklabels(batch_sizes)
    ax.set_xlabel('Batch size')
    ax.set_ylabel('Accuracy')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.grid(axis='y')
    fig.tight_layout()
    root = ospjoin('plots', f'{args.stage}')
    os.makedirs(root, exist_ok=True)
    fig.savefig(ospjoin(root, f'batch_{args.save_name}.{args.extension}'), dpi=300, bbox_inches='tight')


def benchmark_spider(args, all_results, logger):
    """
    For each case and each method, plot the box corresponding to all different runs.
    """
    def check_1(case, method, config_dic):
        """
        Make sure all configs corresponding to 1 case all use the same checkpoint.
        """
        all_archs = [config.getattr('MODEL').getattr('WEIGHTS') for config in config_dic]
        assert len(list(set(all_archs))) == 1, f"Cases should be defined such that all configs in 1 case use the same model.\
                                                Currently, {method} for {case} has {set(all_archs)}." 

    def check_2(case, method, config_dic):
        for c in config_dic:
            assert len(config_dic[c]['all_values']) == 10, f"Case {case} for {method} only contains {len(config_dic[c]['all_values'])} values \
                        for {c.DATASETS.ADAPTATION}"

    benchmark_dic, architectures = get_benchmark_dict(all_results, logger, 'MODEL.WEIGHTS', checks=[check_1, check_2])
    architectures.sort(key=lambda x: -len(x))
    methods = list(benchmark_dic.keys())

    fig = plt.figure(figsize=(20, 13))
    ax = fig.add_subplot(111, polar=True)

    BG_WHITE = "#fbf9f4"
    BLUE = "#2a475e"
    GREY70 = "#b3b3b3"
    GREY_LIGHT = "#f2efe8"

    # The four variables in the plot
    if all([('50' in arch) for arch in architectures]):
        # l0, l1, l2 = 0.4, 0.5, 0.6
        l = [0.4, 0.45, 0.5, 0.55, 0.6]
        angle = 1.1
        checkpoint2pretty['checkpoints/msra/R-50.pkl'] = 'ORIGINAL'
        checkpoint2pretty['checkpoints/pytorch/R-50.pth'] = 'TORCHVISION'
        checkpoint2pretty['checkpoints/simclr/R-50.pth'] = 'SIMCLR'
    else:
        angle = 2.2
        # l0, l1, l2 = 0.45, 0.5, 0.6, 0.75
        l = [0.45, 0.52, 0.6, 0.67, 0.75]
    VARIABLES = [rf'{checkpoint2pretty[arch]}' for arch in architectures]
    VARIABLES_N = len(VARIABLES)

    # The angles at which the values of the numeric variables are placed
    ANGLES = [n / VARIABLES_N * 2 * np.pi for n in range(VARIABLES_N)]
    ANGLES += ANGLES[:1]

    # Padding used to customize the location of the tick labels
    # X_VERTICAL_TICK_PADDING = 5
    # X_HORIZONTAL_TICK_PADDING = 50

    # Angle values going from 0 to 2*pi
    HANGLES = np.linspace(0, 2 * np.pi)

    # Used for the equivalent of horizontal lines in cartesian coordinates plots 
    # The last one is also used to add a fill which acts a background color.
    H = [np.ones(len(HANGLES)) * li for li in l]

    # fig.patch.set_facecolor(BG_WHITE)
    # ax.set_facecolor(BG_WHITE)

    # Rotate the "" 0 degrees on top. 
    # There it where the first variable, avg_bill_length, will go.
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Setting lower limit to negative value reduces overlap
    # for values that are 0 (the minimums)
    ax.set_ylim(l[0] - 0.02, l[-1])

    # Set values for the angular axis (x)
    ax.set_xticks(ANGLES[:-1])
    ax.set_xticklabels(VARIABLES, size=35, y=-0.1)

    # Remove lines for radial axis (y)
    ax.set_yticks([])
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)

    # Remove spines
    ax.spines["start"].set_color("none")
    ax.spines["polar"].set_color("none")

    # Add custom lines for radial axis (y) at 0, 0.5 and 1.
    _ = [ax.plot(HANGLES, h, ls=(0, (6, 6)), c=GREY70) for h in H]

    # Add levels -----------------------------------------------------
    # These labels indicate the values of the radial axis
    PAD = 0.005
    
    size = 20
    _ = [ax.text(angle, li + PAD, f"{int(li * 100)}\%", size=size) for li in l]

    # Now fill the area of the circle with radius 1.
    # This create the effect of gray background.
    ax.fill(HANGLES, H[-1], GREY_LIGHT)

    # Fill lines and dots --------------------------------------------
    per_method_avg = {method: np.round(100 * np.mean([benchmark_dic[method][arch]['mean'] for arch in architectures]), 1) for method in methods}
    sorted_methods = sorted(per_method_avg.items(), key=lambda t: -t[1])
    for idx, (method, perf) in enumerate(sorted_methods):
        values = [benchmark_dic[method][arch]['mean'] for arch in architectures]
        values += values[:1]
        # if per_method_avg[idx] > l0 * 100:
        values = np.clip(values, l[0], l[-1])
        label = f"{method2pretty[method]} ({perf})"
        ax.plot(ANGLES, values, c=method2color[method], linewidth=3, label=rf"\textbf{{{label}}}" if (idx == 0) else label,)
        ax.scatter(ANGLES, values, s=130, c=method2color[method], zorder=10)

    ax.legend(
        loc='center',
        bbox_to_anchor=[1.4, 0.5],       # bottom-right
        ncol=1,
        frameon=False,     # don't put a frame
        prop={'size': 32}
    )

    # ---- Save plots ----

    fig.tight_layout()
    root = ospjoin('plots', f'{args.stage}')
    os.makedirs(root, exist_ok=True)
    fig.savefig(ospjoin(root, f'spider_{args.save_name}.{args.extension}'), dpi=300, bbox_inches='tight')


def save_best_metrics(args, method2bestconfig, logger):
    logger.info(f"Saving best configurations in {args.save_dir}")
    for method in method2bestconfig:
        if method != 'NonAdaptiveMethod':
            best_config = method2bestconfig[method]['best_config']
            root = ospjoin('configs', 'method', 'adaptation', 'best', method)
            os.makedirs(root, exist_ok=True)
            with open(ospjoin(root, f'{args.save_name}.yaml'), 'w') as file:
                yaml.dump(best_config.todict(), file)


if __name__ == "__main__":
    logger = setup_logger(name=__name__)
    args = parse_args()

    if args.latex:
        logger.info("Activating latex")
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

        plt.rcParams.update({'font.size': 26})

    all_results = read_all_results(args, logger)

    if args.action == 'benchmark_box':
        benchmark_box(args, all_results, logger)

    if args.action == 'benchmark_batch':
        benchmark_batch(args, all_results, logger)

    if args.action == 'benchmark_spider':
        benchmark_spider(args, all_results, logger)

    if args.action == 'log_best':
        method2bestconfig = get_per_method_best_config(all_results, logger)
        log_best_metrics(method2bestconfig, logger)
        if args.save:
            save_best_metrics(args, method2bestconfig, logger)

    elif args.action == 'spider_chart':
        baseline_results, adaptive_results, hparams2tune, hparams_values = get_per_case_best_config(args, all_results, logger)
        hparams_spider_chart(args, baseline_results, adaptive_results, hparams2tune, hparams_values, logger)

    elif args.action == 'cross_cases':
        baseline_results, adaptive_results, hparams2tune, hparams_values = get_per_case_best_config(args, all_results, logger)
        make_cross_cases_heatmap(args, adaptive_results, baseline_results, all_results, logger)


