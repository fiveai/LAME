import torch
import torch.jit
import logging
from typing import List, Dict

from .adaptive import AdaptiveMethod
from .build import ADAPTER_REGISTRY

__all__ = ["Oracle"]

logger = logging.getLogger(__name__)


@ADAPTER_REGISTRY.register()
class Oracle(AdaptiveMethod):
    """
    An oracle that uses the target labels to train online.
    """  
    def __init__(self, configs, args, **kwargs):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__(configs, args, **kwargs)

    def run_optim_step(self, batched_inputs: List[Dict[str, torch.Tensor]], **kwargs):

        prob = self.model(batched_inputs)['probas']
        gt = torch.tensor([obj["instances"].gt_classes[0] for obj in batched_inputs]).to(prob.device)

        hot_labels = self.one_hot(gt, self.num_classes)
        loss = - (hot_labels * torch.log(prob + 1e-10)).sum(-1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.metric_hook.scalar_dic['full_loss'].append(loss.item())