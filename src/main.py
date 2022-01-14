import os
import numpy as np
import time
import logging
import argparse
import itertools
from src.utils.logger import setup_logger
from os.path import join, exists
from src.utils.plot import main as plot_main
from pathlib import Path
from typing import Optional
from src.adaptation.build import build_adapter
from src.config import get_cfg
import shutil
from os.path import join as ospjoin

logger = setup_logger(distributed_rank=0, name="GPU allocation")


def maybe_debug_mode(cfg):
    """
    Changes some params in config if debug mode is activated.
    """
    logger = logging.getLogger("preparation")

    # --- DEBUG MODE ---
    if cfg.DEBUG:
        
        logger.warning("Entering DEBUG mode")
        cfg.SAVE_PLOTS = True
        cfg.VIS_PERIOD = 1

        cfg.ADAPTATION.MAX_VISU_FRAMES = 100
        cfg.ADAPTATION.MAX_BATCH_PER_EPISODE = 15
        cfg.ADAPTATION.VISU_PERIOD = 1
        cfg.ADAPTATION.BATCH_SIZE = 10

        cfg.DATASETS.MAX_DATASET_SIZE = 10000


def need_do_exp(cfg, current_outdir: Optional[str] = None) -> bool:
    """
    Assesses whether the experiment needs to be done or not.
    """

    do_exp = True
    if current_outdir is None:
        current_outdir = cfg.OUTPUT_DIR
    logger = logging.getLogger("preparation")
    log_file = join(current_outdir, "log.txt")
    dummy_final_file = join(current_outdir, "dummy_final.txt")

    if exists(current_outdir):
        if exists(dummy_final_file):
            if cfg.OVERRIDE:
                logger.warning(f"{current_outdir} already exists, \n) \
                            But overriding option is activated")
                clean_folder(current_outdir)
            else:
                logger.warning(f"{current_outdir} already exists. \n) \
                                Besides, corresponding experiment was properly completed, \n\
                                Thus not doing the current exp.")
                do_exp = False

        elif exists(log_file) and (((time.time() - os.stat(log_file).st_mtime) / 60) < 1):
            if cfg.OVERRIDE:
                logger.warning(f"{current_outdir} already exists, \n) \
                            But overriding option is activated")
                clean_folder(current_outdir)
            else:
                logger.warning(f"{current_outdir} already exists. \n \
                                Besides, the experiment seems to be already running, \n\
                                Thus not doing the current exp.")
                do_exp = False            

        elif exists(join(current_outdir, "tensorboard")):
            logger.warning(f"{current_outdir} already exists. \n \
                             However, previous experiments was not properly completed, \n \
                             Thus overwritting.")
            clean_folder(current_outdir)

    return do_exp


def clean_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def main_adaptation(args):
    """
    Main adaptation entry point.
    """

    # Validation
    if args.mode == 'validation':
        cfg, do_exp = get_config_from_args(args)
        params2tune = cfg.ADAPTATION.HPARAMS2TUNE
        values = cfg.ADAPTATION.HPARAMS_VALUES
        init_output_dir = cfg.OUTPUT_DIR
        for i, combination in enumerate(itertools.product(*values)):
            new_output_dir = ospjoin(init_output_dir, str(combination))
            new_opt = list(zip(params2tune, combination))
            new_opt = [item for sublist in new_opt for item in sublist]
            cfg.merge_from_list(new_opt)
            cfg.merge_from_list(['OUTPUT_DIR', new_output_dir])
            do_exp = need_do_exp(cfg)
            if do_exp:
                adapter = build_adapter(cfg, args, setup_logger=(i == 0))
                adapter.run_full_adaptation()
                plot_main(selected_folders=[Path(new_output_dir)], save_path=Path(new_output_dir), folder='numpy',
                          force_labels=False, labels=[], dpi=200, out_extension='png', fontsize=11, max_columns=2)
                # shutil.rmtree(Path(new_output_dir) / 'numpy')
                np_files = (Path(new_output_dir) / 'numpy').glob('*.npy')
                for file in np_files:
                    if file.stem != 'accuracy':
                        os.remove(file)
                with open(os.path.join(new_output_dir, "dummy_final.txt"), "w") as f:
                    f.write("Experiment completed")
    else:
        cfg, do_exp = get_config_from_args(args)
        # Just to show other processes this GPU is being used
        if do_exp:
            adapter = build_adapter(cfg, args)
            adapter.run_full_adaptation()
            out_dir = cfg.OUTPUT_DIR
            plot_main(selected_folders=[Path(out_dir)], save_path=Path(out_dir), folder='numpy', force_labels=False,
                      labels=[], dpi=200, out_extension='png', fontsize=11, max_columns=2)
            with open(os.path.join(cfg.OUTPUT_DIR, "dummy_final.txt"), "w") as f:
                f.write("Experiment completed")


def get_config_from_args(args):
    """
    Create configs and perform basic setups.
    """
    logger = setup_logger(name="Preparation")
    cfg = get_cfg()
    cfg.mode = args.mode
    cfg.merge_from_file(args.model_config)
    cfg.merge_from_file(args.data_config)
    cfg.merge_from_file(args.method_config)
    cfg.merge_from_list(args.opts)
    cfg.index = 0
    if args.mode == 'benchmark':
        method = cfg.ADAPTATION.METHOD
        if method != 'NonAdaptiveMethod':
            best_conf_path = ospjoin('configs', 'method', 'best', method, 'overall_best.yaml')
            logger.info(f"Loading best config from {best_conf_path}")
            cfg.merge_from_file(best_conf_path)

    do_exp = need_do_exp(cfg, cfg.OUTPUT_DIR)
    if do_exp:
        maybe_debug_mode(cfg)
    return cfg, do_exp


def argument_parser(epilog=None):
    """
    Create a parser with some common arguments used by src users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-config", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--model-config", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--method-config", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--mode", required=True, help="benchmark or validation")
    parser.add_argument("--allowed_gpus", type=str, required=True,
                        help="Which GPUs to use on the cluster. Either set to 'all' or e.g. 0,1,2")
    parser.add_argument("--gpu_per_exp", type=str, default=1,
                        help="How many GPUs are  required for each exp.")
    parser.add_argument("--gpu_free_under", type=str, default=20,
                        help="When a GPU is using less gpu_free_under memory, it is considered free.")
    parser.add_argument(
        "opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. "
        "See config references at "
        "https://src.readthedocs.io/modules/config.html#config-references",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def get_free_gpus(args):
    """
    Works for nvidia GPUS only !
    """
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total = np.array([int(x.split(',')[0]) for x in devices_info])
    used = np.array([int(x.split(',')[1]) for x in devices_info])
    mask = (used < args.gpu_free_under)
    free = np.where(mask)[0]
    return free, total[mask], used[mask]


def set_env(ids: np.ndarray):
    # Very important to keep thesel ines in this exact order
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in ids])


def book_memory(ids: np.ndarray, total: np.ndarray, used: np.ndarray):
    for i in range(torch.cuda.device_count()):
        # max_mem = int(total[i] * 0.6)
        # block_mem = max_mem - used[i]
        x = torch.FloatTensor(256, 1024, 20).to(i)
        del x


if __name__ == "__main__":
    args = argument_parser().parse_args()
    gpu_required_by_exp = args.gpu_per_exp

    while True:
        free_ids, total, used = get_free_gpus(args)
        if args.allowed_gpus == 'all':
            authorized_gpus = free_ids
        else:
            authorized_gpus = set([int(x) for x in args.allowed_gpus.split(',')])
        logger.info(f"Remaining free and allowed GPUs:  {authorized_gpus}")
        free_and_authorized = list(set(free_ids).intersection(authorized_gpus))
        if len(free_and_authorized) >= gpu_required_by_exp:
            selected_gpus = free_and_authorized[:gpu_required_by_exp]
            break
        else:
            logger.info("Waiting for some GPUs to free up.")
            time.sleep(100)

        # assert all([(x in free_ids) for x in selected_gpus]), "Some GPUs provided by user were not free."
    set_env(selected_gpus)
    import torch  # !!!! Needs to come after set_env(), because it seals once and for all the GPUs that will be used
    book_memory(selected_gpus, total, used)
    logger.info(f"GPUs {selected_gpus} allocated.")
    print("Command Line Args:", args)
    from src.engine import launch
    launch(
        main_adaptation,
        num_gpus_per_machine=1,
        num_machines=1,
        machine_rank=0,
        args=(args,),
    )