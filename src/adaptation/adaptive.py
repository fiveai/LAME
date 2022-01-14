
from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit
import logging
from typing import List, Dict
 
from tqdm import tqdm
from contextlib import ExitStack, contextmanager
import time
from functools import partial

from src.utils.events import EventStorage
from src.data.catalog import DatasetCatalog

from .non_adaptive import NonAdaptiveMethod
from .build import ADAPTER_REGISTRY
__all__ = ["AdaptiveMethod",
           "collect_params",
           "configure_model"]
logger = logging.getLogger(__name__)


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """

    # Important to do it module-wise for our purpose
    training_modes = {}
    for named_m, module in model.named_modules():
        training_modes[named_m] = module.training
    model.eval()
    yield
    for named_m, module in model.named_modules():
        module.training = training_modes[named_m]


@ADAPTER_REGISTRY.register()
class AdaptiveMethod(NonAdaptiveMethod):
    """
    A class specifically designed for methods that modify the network by optimizing some loss in an
    online fashion.
    """
    def __init__(self, cfg, args, **kwargs):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__(cfg, args, **kwargs)
        data_names = cfg.DATASETS.ADAPTATION[0]
        self.num_classes = len(DatasetCatalog.get(data_names).thing_classes)
        self.steps = cfg.ADAPTATION.STEPS
        self.online = cfg.ADAPTATION.ONLINE
        partition: dict = self.model.backbone.partition_parameters(granularity=cfg.MODEL.PARTITION_GRANULARITY)
        all_params, all_modules, all_param_names = collect_params(partition,
                                                                  cfg.ADAPTATION.PARAMS2ADAPT)
        self.logger.info(f"Optimizing {all_param_names}")
        if cfg.ADAPTATION.OPTIMIZER == 'Adam':
            cfg.ADAPTATION.LR
            opt = torch.optim.Adam(all_params,
                                   lr=cfg.ADAPTATION.LR,
                                   weight_decay=cfg.ADAPTATION.WEIGHT_DECAY)
        else:
            opt = torch.optim.SGD(all_params,
                                  lr=cfg.ADAPTATION.LR,
                                  weight_decay=cfg.ADAPTATION.WEIGHT_DECAY,
                                  momentum=cfg.ADAPTATION.OPTIM_MOMENTUM,
                                  nesterov=cfg.ADAPTATION.NESTEROV and cfg.ADAPTATION.OPTIM_MOMENTUM)
        self.optimizer = opt
        self.model_config_fn = partial(configure_model,
                                       parameters=all_params,
                                       modules=all_modules,
                                       state_dict=deepcopy(self.model.state_dict()))
        self.opt_state_dict = deepcopy(opt.state_dict())

    def reset_model_optim(self, **kwargs):
        self.model_config_fn(self.model)
        self.optimizer.load_state_dict(self.opt_state_dict)

    def run_episode(self, loader: torch.utils.data.DataLoader) -> EventStorage:
        """
        Loader contains all the samples in one run, and yields them by batches.
        """
        self.reset_model_optim()
        max_inner_iters = self.cfg.ADAPTATION.MAX_BATCH_PER_EPISODE
        with EventStorage(0) as local_storage:

            for _ in range(self.steps):
                batch_limitator = range(max_inner_iters)
                bar = tqdm(loader, total=min(len(loader), max_inner_iters))
                bar.set_description(f"Running optimization steps {'online' if self.online else 'offline'}")
                for i, (batched_inputs, indexes) in zip(batch_limitator, bar):

                    # --- Optimization part ---

                    self.run_optim_step(batched_inputs, indexes=indexes, loader=loader, batch_index=i)

                    # --- Evaluation part ---

                    if self.online:
                        with ExitStack() as stack:
                            stack.enter_context(inference_context(self.model))
                            stack.enter_context(torch.no_grad())
                            self.before_step()
                            t0 = time.time()
                            outputs = self.run_step(batched_inputs)
                            self.metric_hook.scalar_dic["inference_time"].append(time.time() - t0)
                            self.after_step(batched_inputs, outputs)

        return local_storage

    def run_optim_step(self, batched_inputs: List[Dict[str, torch.Tensor]], **kwargs):
        raise NotImplementedError


def collect_params(partition, params2adapt: str):
    """
    Collect parameters using the params2adapt list of layers.
    """
    params = []
    param_names = []
    modules = []

    if len(params2adapt):
        section_param2adapt = params2adapt.split("_")
        section2adapt = [t.split('-')[0] for t in section_param2adapt]
        params2adapt = [t.split('-')[1] for t in section_param2adapt]
        if section2adapt[0] == 'all':
            section2adapt = list(partition.keys())
        else:
            section2adapt = [int(x) for x in section2adapt]

        for section, param_type in zip(section2adapt, params2adapt):
            if param_type == 'all':
                param_types = list(partition[section].keys())
            else:
                param_types = [param_type]

            for p_type in param_types:
                param_names += partition[section][p_type]['names']
                params += partition[section][p_type]['parameters']
                modules += partition[section][p_type]['modules']
    return params, modules, param_names

        
def configure_model(model, parameters, modules, state_dict):

    assert len(parameters) == len(modules)

    # --- Reset parameters ---
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)

    # --- Put in proper mode only the right modules ---

    model.eval()
    for p, m in zip(parameters, modules):
        if not isinstance(m, nn.Dropout):
            m.training = True

    # --- Activate grad for useful parameters ---

    model.requires_grad_(False)
    for p in parameters:
        p.requires_grad_(True)
