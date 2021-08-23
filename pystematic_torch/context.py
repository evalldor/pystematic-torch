import logging

import torch

from .recording import Recorder

logger = logging.getLogger('pystematic.torch')



class TorchContext:

    def cuda(self):
        """Moves the context to a local cuda device.
        """
        raise NotImplementedError()

    def cpu(self):
        """Moves the context to the cpu.
        """
        raise NotImplementedError()

    def ddp(self):
        """Moves the context to a distributed data-parallell setting. Can only
        be used in torch.distributed is initialized.
        """
        raise NotImplementedError()

    def state_dict(self):
        """Returns the whole state of the context by iterating all registered
        items and calling ``state_dict()`` on the item to retrieve its state.
        Primitive values will also be saved.
        """
        raise NotImplementedError()

    def load_state_dict(self, state):
        """Sets the state for the context.

        Args:
            state (dict, list): The state to load.
        """
        raise NotImplementedError()

    def autotransform(self):
        """Transforms the context according to the current experiment
        parameters. More specifically it; loads a state_dict from the parameter
        'checkpoint' if set, moves to cuda if paramter 'cuda' is set, moves to
        distributed if parameter 'distributed' is set.
        """
        from pystematic import params
        if params["checkpoint"]:
            logger.info(f"Loading checkpoint '{params['checkpoint']}'.")
            with open(params["checkpoint"], "rb") as f:
                self.load_state_dict(torch.load(f, map_location="cpu"))

        if params["cuda"]:
            self.cuda()

        if params["distributed"]:
            self.ddp()
 
    def _wrap_value(self, value):
        if isinstance(value, (list, tuple)):
            new_value = ContextList()
            for val in value:
                new_value.append(val)

            return new_value
        elif isinstance(value, dict):
            new_value = ContextDict()
            for key, val in value.items():
                new_value[key] = val
            
            return new_value

        return value

    def _to_cuda(self, item):
        return self._move_to_device(item, f"cuda:{torch.cuda.current_device()}")

    def _to_cpu(self, item):
        return self._move_to_device(item, "cpu")

    def _to_ddp(self, name, item):
        if isinstance(item, TorchContext):
            item.ddp()

        elif isinstance(item, torch.nn.Module):
            if any([p.requires_grad for p in item.parameters()]):
                logger.debug(f"Converting to distributed for model '{name}'.")

                item = torch.nn.parallel.DistributedDataParallel(
                    module=item,
                    device_ids=[torch.cuda.current_device()]
                )

        elif isinstance(item, Recorder):
            if torch.distributed.get_rank() != 0: # Only rank zero may log stats
                item.silence()
                logger.debug(f"Silencing recorder '{name}' in rank '{torch.distributed.get_rank()}'.")

        return item
    
    def _get_state_dict(self, item):
        supported_types = (int, float, complex, str)
        
        if callable(getattr(item, "state_dict", None)):
            if isinstance(item, torch.nn.parallel.DistributedDataParallel):
                return item.module.state_dict()
            else:
                return item.state_dict()

        elif isinstance(item, supported_types):
            return {
                "native_value": item
            }
        else:
            logger.debug(f"Cannot checkpoint object of type '{type(item)}'")

        return None

    def _set_state_dict(self, item, state_dict):
        
        supported_types = (int, float, complex, str)
        if callable(getattr(item, "load_state_dict", None)):
            if isinstance(item, torch.nn.parallel.DistributedDataParallel):
                item.module.load_state_dict(self._move_to_same_device_as(state_dict, item.module))
            else:
                item.load_state_dict(self._move_to_same_device_as(state_dict, item))

        elif isinstance(item, supported_types):
            return state_dict["native_value"] if "native_value" in state_dict else state_dict["count"]
            
        else:
            logger.debug(f"Cannot checkpoint object of type '{type(item)}'.")

        return item

    def _move_to_same_device_as(self, to_move, target):
        if hasattr(target, "device"):
            return self._move_to_device(to_move, target.device)

        elif callable(getattr(target, "parameters", None)):
            try:
                return self._move_to_device(to_move, next(target.parameters()).device)
            except StopIteration:
                pass

        return to_move

    def _move_to_device(self, obj, device):
        if isinstance(obj, dict):
            res = {}
            for name, value in obj.items():
                res[name] = self._move_to_device(value, device)

        elif isinstance(obj, (list, tuple)):
            res = []
            for i in range(len(obj)):
                res.append(self._move_to_device(obj[i], device))

        elif callable(getattr(obj, "to", None)):
            res = obj.to(device=device)
        elif isinstance(obj, torch.optim.Optimizer):
            obj.load_state_dict(self._move_to_device(obj.state_dict(), device))
            res = obj
        else:
            res = obj
            # raise Exception(f"Unsupported object type '{type(obj)}'")

        return res


def _move_to_same_device_as(to_move, target):
    if hasattr(target, "device"):
        return _move_to_device(to_move, target.device)

    elif callable(getattr(target, "parameters", None)): # torch modules
        try:
            return _move_to_device(to_move, next(target.parameters()).device)
        except StopIteration:
            pass

    return to_move


def _move_to_device(obj, device):
    if callable(getattr(obj, "to", None)):
        return obj.to(device=device)

    if isinstance(obj, torch.optim.Optimizer):
        obj.load_state_dict(_move_to_device(obj.state_dict(), device))
        return obj

    if isinstance(obj, dict):
        res = {}
        for name, value in obj.items():
            res[name] = _move_to_device(value, device)
        
        return res

    if isinstance(obj, (list, tuple)):
        res = []
        for i, sub_item in enumerate(obj):
            res.append(_move_to_device(sub_item, device))
        
        return res

    return obj


def _get_state_dict(item):

    if callable(getattr(item, "state_dict", None)):
        if isinstance(item, torch.nn.parallel.DistributedDataParallel):
            return item.module.state_dict()
        else:
            return item.state_dict()
    
    if isinstance(item, (int, float, complex, str)):
        return item

    if isinstance(item, dict):
        res = {}

        for name, sub_item in item.items():
            res[name] = _get_state_dict(sub_item)

        return res

    if isinstance(item, (list, tuple)):
        res = []

        for sub_item in item:
            res.append(_get_state_dict(sub_item))

        return res

    return None


def _set_state_dict(item, state, path=[]):

    if callable(getattr(item, "load_state_dict", None)):
        if isinstance(item, torch.nn.parallel.DistributedDataParallel):
            item.module.load_state_dict(_move_to_same_device_as(state, item.module))
        else:
            item.load_state_dict(_move_to_same_device_as(state, item))
        
        return item

    if isinstance(item, (int, float, complex, str)):
        if isinstance(state, (int, float, complex, str)):
            raise ValueError(f"Error when setting state for item '{'.'.join(path)}', "
                             f"expected a primitive value, got '{type(state)}'.")

        return state

    if isinstance(item, dict):
        if not isinstance(state, dict):
            raise ValueError(f"Error when setting state for item '{'.'.join(path)}', "
                             f"expected a dict, got '{type(state)}'.")

        res = {}

        for name, sub_item in item.items():
            if name in state:
                res[name] = _set_state_dict(sub_item, state[name], path + [name])
            else:
                raise ValueError(f"Error when setting state for item '{'.'.join(path)}', "
                                 f"key '{name}' was not found in state.")

        return res

    if isinstance(item, (list, tuple)):
        if not isinstance(state, (list, tuple)):
            raise ValueError(f"Error when setting state for item '{'.'.join(path)}', "
                             f"expected a list, got '{type(state)}'")

        if len(item) != len(state):
            raise ValueError(f"Error when setting state for item '{'.'.join(path)}', "
                             f"expected a list of length '{len(item)}', got one of length '{len(state)}'.")
        
        res = []

        for i, sub_item in enumerate(item):
            res.append(_set_state_dict(sub_item, state[i], path + [str(i)]))

        return res

    if state is not None:
        raise ValueError(f"Error when setting state for item '{'.'.join(path)}', "
                         f"expected None, got '{type(state)}'")

    return item


def _to_distributed_data_parallel(item):
        if callable(getattr(item, "ddp", None)):
            return item.ddp()

        if isinstance(item, torch.nn.Module):
            if any([p.requires_grad for p in item.parameters()]):
                logger.debug(f"Converting to distributed for model '{item}'.")

                return torch.nn.parallel.DistributedDataParallel(
                    module=item,
                    device_ids=[torch.cuda.current_device()]
                )
            
            return item

        if isinstance(item, Recorder):
            if torch.distributed.get_rank() != 0: # Only rank zero may log stats
                item.silence()
                logger.debug(f"Silencing recorder '{item}' in rank '{torch.distributed.get_rank()}'.")

            return item

        if isinstance(item, dict):
            res = {}
            for name, sub_item in item.items():
                res[name] = _to_distributed_data_parallel(sub_item)
            
            return res

        if isinstance(item, (list, tuple)):
            res = []
            for i, sub_item in enumerate(item):
                res.append(_to_distributed_data_parallel(sub_item))
            
            return res

        return item


class ContextObject(TorchContext):

    def __init__(self):
        object.__setattr__(self, "_items", {})

    def __getattr__(self, name):
        if name in self._items:
            return self._items[name]

        raise AttributeError(f"There's no attribute named '{name}'.")

    def __setattr__(self, name, value):
        self._items[name] = self._wrap_value(value)

    def has(self, name : str):
        return name in self._items

    def cuda(self):
        for name, item in self._items.items():
            self._items[name] = self._to_cuda(item)

        return self
  
    def cpu(self):
        for name, item in self._items.items():
            self._items[name] = self._to_cpu(item)
        
        return self

    def ddp(self):
        assert torch.distributed.is_initialized(), "You must initialize a distributed runtime before calling ddp."
        
        for name, item in self._items.items():
            self._items[name] = self._to_ddp(name, item)
        
        return self
        
    def state_dict(self) -> dict:
        dict_with_state = {}

        for name, item in self._items.items():
            dict_with_state[name] = self._get_state_dict(item)

        return dict_with_state

    def load_state_dict(self, state : dict) -> None:

        for name, item_state in state.items():
            if name in self._items:
                self._items[name] = self._set_state_dict(self._items[name], item_state)
            

class ContextDict(TorchContext):
    
    def __init__(self):
        object.__setattr__(self, "_items", {})

    def __getitem__(self, name):
        return self._items[name]

    def __setitem__(self, name, value):
        self._items[name] = self._wrap_value(value)

    def items(self):
        return {key:item for key, item in self._items}

    def keys(self):
        return self._items.keys()

    def values(self):
        return self._items.values()

    def cuda(self):
        for name, item in self._items.items():
            self._items[name] = self._to_cuda(item)

        return self
  
    def cpu(self):
        for name, item in self._items.items():
            self._items[name] = self._to_cpu(item)
        
        return self

    def ddp(self):
        assert torch.distributed.is_initialized(), "You must initialize a distributed runtime before calling ddp."
        
        for name, item in self._items.items():
            self._items[name] = self._to_ddp(name, item)
        
        return self

    def state_dict(self) -> dict:
        dict_with_state = {}

        for name, item in self._items.items():
            dict_with_state[name] = self._get_state_dict(item)

        return dict_with_state

    def load_state_dict(self, state : dict) -> None:

        for name, item in state.items():
            if name in self._items:
                self._items[name] = self._set_state_dict(self._items[name], item)
            

class ContextList(TorchContext):
    
    def __init__(self):
        object.__setattr__(self, "_items", [])

    def __getitem__(self, index):
        return self._items[index]

    def __setitem__(self, name, value):
        self._items[name] = self._wrap_value(value)

    def insert(self, index, value):
        self._items.insert(index, self._wrap_value(value))

    def append(self, value):
        self._items.append(self._wrap_value(value))

    def __len__(self):
        return len(self._items)
    
    def cuda(self):
        for i, item in enumerate(self._items):
            self._items[i] = self._to_cuda(item)
        
        return self
  
    def cpu(self):
        for i, item in enumerate(self._items):
            self._items[i] = self._to_cpu(item)
        
        return self

    def ddp(self):
        assert torch.distributed.is_initialized(), "You must initialize a distributed runtime before calling ddp."
        
        for i, item in enumerate(self._items):
            self._items[i] = self._to_ddp(item)
        
        return self
    
    def state_dict(self) -> list:
        list_with_state = []

        for i, item in enumerate(self._items):
            list_with_state.append(self._get_state_dict(item))

        return list_with_state

    def load_state_dict(self, state : list) -> None:

        for i, item in enumerate(state):
            self._items[i] = self._set_state_dict(self._items[i], item)


class ContextObject2:
   
    def state_dict(self) -> dict:
        """Returns the whole state of the context by iterating all registered
        items and calling ``state_dict()`` on the item to retrieve its state.
        Primitive values will also be saved.
        """
        state = {}

        for name, item in vars(self).items():
            state[name] = _get_state_dict(item)

        return state

    def load_state_dict(self, state : dict) -> None:
        """Sets the state for the context.

        Args:
            state (dict, list): The state to load.
        """
        for name, item_state in state.items():
            if name in vars(self):
                setattr(self, name, _set_state_dict(getattr(self, name), item_state))

    def to(self, device):
        """Move the context to a specific device

        Args:
            device (str, torch.Device): The device to move the context to.
        """
        for name, item in vars(self).items():
            setattr(self, name, _move_to_device(item, device))
        
        return self

    def cuda(self):
        """Moves the context to a local cuda device.
        """
        return self.to(f"cuda:{torch.cuda.current_device()}")

    def cpu(self):
        """Moves the context to the cpu.
        """
        return self.to("cpu")

    def ddp(self):
        """Moves the context to a distributed data-parallell setting. Can only
        be used in torch.distributed is initialized.
        """
        for name, item in vars(self).items():
            setattr(self, name, _to_distributed_data_parallel(item))
        
        return self

    def autotransform(self):
        """Transforms the context according to the current experiment
        parameters. More specifically it; loads a state_dict from the parameter
        'checkpoint' if set, moves to cuda if paramter 'cuda' is set, moves to
        distributed if parameter 'distributed' is set.
        """
        from pystematic import params

        if params["checkpoint"]:
            logger.info(f"Loading checkpoint '{params['checkpoint']}'.")
            with open(params["checkpoint"], "rb") as f:
                self.load_state_dict(torch.load(f, map_location="cpu"))

        if params["cuda"]:
            self.cuda()

        if params["distributed"]:
            self.ddp()
        
        return self
