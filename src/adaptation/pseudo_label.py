import torch
import torch.jit
import logging
from typing import List, Dict

from .adaptive import AdaptiveMethod
from .build import ADAPTER_REGISTRY
import torch.nn.functional as F
__all__ = ["PseudoLabeller"]

logger = logging.getLogger(__name__)


@ADAPTER_REGISTRY.register()
class PseudoLabeller(AdaptiveMethod):

    def __init__(self, cfg, args, **kwargs):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__(cfg, args, **kwargs)
        self.threshold = self.cfg.ADAPTATION.PL_THRESHOLD

    def run_optim_step(self, batched_inputs: List[Dict[str, torch.Tensor]], **kwargs):
        
        prob = self.model(batched_inputs)['probas']

        max_prob = torch.max(prob.detach(), dim=-1).values
        mask = max_prob > self.threshold

        if mask.sum():
            hot_labels = F.one_hot(prob[mask].argmax(-1).detach(), self.num_classes)
            loss = - (hot_labels * torch.log(prob[mask] + 1e-10)).sum(-1).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.metric_hook.scalar_dic["nb_pseudo_labelled"].append(mask.sum().item() / mask.size(0))