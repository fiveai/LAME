import torch
import torch.jit
import logging
from typing import List, Dict
import time

from .adaptive import AdaptiveMethod
from .build import ADAPTER_REGISTRY

__all__ = ["Tent"]

logger = logging.getLogger(__name__)


@ADAPTER_REGISTRY.register()
class Tent(AdaptiveMethod):
    """
    Tent method https://arxiv.org/abs/2006.10726.
    """

    def run_optim_step(self, batched_inputs: List[Dict[str, torch.Tensor]], **kwargs):

        t0 = time.time()
        probas = self.model(batched_inputs)['probas']
        self.metric_hook.scalar_dic["forward_time"].append(time.time() - t0)
        t1 = time.time()

        log_probas = torch.log(probas + 1e-10)
        entropy = -(probas * log_probas).sum(-1).mean(0)
        loss = entropy

        self.optimizer.zero_grad()
        loss.backward()  # type: ignore[union-attr]
        self.optimizer.step()

        self.metric_hook.scalar_dic["optimization_time"].append(time.time() - t1)
        self.metric_hook.scalar_dic['full_loss'].append(loss.item())