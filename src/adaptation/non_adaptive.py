import torch
import torch.jit
import logging
from typing import List, Dict

from .adapter import DefaultAdapter
from .build import ADAPTER_REGISTRY
from src.utils.events import EventStorage
from tqdm import tqdm
__all__ = ["NonAdaptiveMethod"]

logger = logging.getLogger(__name__)
torch.multiprocessing.set_sharing_strategy('file_system')


@ADAPTER_REGISTRY.register()
class NonAdaptiveMethod(DefaultAdapter):
    """
    A class for the Baseline method that adapts nothing and just predicts on the fly.
    """  
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.all_classes = []
        self.model.eval()

    def run_episode(self, loader: torch.utils.data.DataLoader) -> EventStorage:

        with EventStorage(0) as local_storage:
            bar = tqdm(loader)
            batch_limitator = range(self.cfg.ADAPTATION.MAX_BATCH_PER_EPISODE)
            for i, (batched_inputs, indexes) in zip(batch_limitator, bar):
                with torch.no_grad():
                    self.before_step()
                    outputs = self.run_step(batched_inputs)
                    self.after_step(batched_inputs, outputs)

        return local_storage

    def run_step(self, batched_inputs: List[Dict[str, torch.Tensor]]):

        out = self.model(batched_inputs)
        probas = out["probas"]
        final_output = self.model.format_result(batched_inputs, probas)

        return final_output