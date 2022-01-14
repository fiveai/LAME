
import torch

from src.utils.logger import _log_api_usage
from src.utils.registry import Registry

MAPPER_REGISTRY = Registry("MAPPER")
MAPPER_REGISTRY.__doc__ = """
Registry for class mappers. Useful for the finetuning stage.
"""


def build_mapper(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    mapper_name = cfg.DATASETS.MAPPER.NAME
    mapper = MAPPER_REGISTRY.get(mapper_name)(cfg)
    _log_api_usage("mapper." + mapper_name)
    return mapper