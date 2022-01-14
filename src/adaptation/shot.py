import torch
import torch.jit
import logging
import time
from typing import List, Dict
from .adaptive import AdaptiveMethod
from .build import ADAPTER_REGISTRY
__all__ = ["Shot"]

logger = logging.getLogger(__name__)


@ADAPTER_REGISTRY.register()
class Shot(AdaptiveMethod):
    """
    Shot-IM method https://arxiv.org/abs/2002.08546.
    """

    def run_optim_step(self, batched_inputs: List[Dict[str, torch.Tensor]], **kwargs):

        t0 = time.time()
        res = self.model(batched_inputs)

        t1 = time.time()
        self.metric_hook.scalar_dic["forward_time"].append(t1 - t0)

        probas = res['probas']

        softmax_out = probas
        msoftmax = softmax_out.mean(0)

        cond_ent = - (softmax_out * torch.log(softmax_out + 1e-10)).sum(-1).mean(0)
        ent = - (msoftmax * torch.log(msoftmax + 1e-10)).sum(-1)

        classifier_loss = cond_ent - ent

        self.optimizer.zero_grad()
        classifier_loss.backward()
        self.optimizer.step()

        self.metric_hook.scalar_dic["optimization_time"].append(time.time() - t1)
        self.metric_hook.scalar_dic['ent'].append(ent.item())
