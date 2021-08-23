import importlib_metadata

__version__ = importlib_metadata.version(__name__)

from .torch_plugin import TorchPlugin