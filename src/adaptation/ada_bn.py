import torch
import torch.jit
import logging
from typing import List, Dict
from .adaptive import AdaptiveMethod
from .build import ADAPTER_REGISTRY
__all__ = ["AdaBN"]

logger = logging.getLogger(__name__)


@ADAPTER_REGISTRY.register()
class AdaBN(AdaptiveMethod):
    """

    AdaBN method https://arxiv.org/abs/1603.04779 adapted for online purposed.

    """  
     
    def run_optim_step(self, batched_inputs: List[Dict[str, torch.Tensor]], **kwargs):

        with torch.no_grad():
            _ = self.model(batched_inputs)['probas']