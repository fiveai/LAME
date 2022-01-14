from src.utils.registry import Registry

ADAPTER_REGISTRY = Registry("ADAPTER")  # noqa F401 isort:skip
ADAPTER_REGISTRY.__doc__ = """
Registry for methods.
"""


def build_adapter(cfg, args, **kwargs):
    """
    Build the adapter module, which is basically the method that will be used.
    All methods are available in the ADAPTER registry.
    """
    if "setup_logger" not in kwargs:
        kwargs["setup_logger"] = True
    METHOD = cfg.ADAPTATION.METHOD
    method = ADAPTER_REGISTRY.get(METHOD)(cfg, args, **kwargs)
    return method
