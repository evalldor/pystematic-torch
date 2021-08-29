import logging

import torch
import wrapt
import tqdm

from .recording import Recorder

logger = logging.getLogger('pystematic.torch')


class DDPModuleProxy(wrapt.ObjectProxy):
    """Delegates any unknown getattr calls to the underlying module. Makes the
    DDP module completely transparent.
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.__wrapped__.module, name)

    def __call__(self, *args, **kwargs):
        return self.__wrapped__(*args, **kwargs)


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
        if not isinstance(state, (int, float, complex, str)):
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

            return DDPModuleProxy(torch.nn.parallel.DistributedDataParallel(
                module=item,
                device_ids=[torch.cuda.current_device()]
            ))
        
        return item

    if isinstance(item, Recorder):
        if torch.distributed.get_rank() != 0: # Only rank zero may log stats
            item.silence()
            logger.debug(f"Silencing recorder '{item}' in rank '{torch.distributed.get_rank()}'.")

        return item

    if isinstance(item, dict):
        return {name: _to_distributed_data_parallel(sub_item) for name, sub_item in item.items()}

    if isinstance(item, (list, tuple)):
        return [_to_distributed_data_parallel(sub_item) for sub_item in item]

    return item


class Context:

    def state_dict(self) -> dict:
        """Returns the whole state of the context by iterating all registered
        items and calling ``state_dict()`` on the item to retrieve its state.
        Primitive values will also be saved.
        

        Returns:
            dict: A dict representing the state of all registered objects.
        """

        
        return {name: _get_state_dict(item) for name, item in vars(self).items()}

    def load_state_dict(self, state : dict) -> None:
        """Sets the state for the context.

        Args:
            state (dict): The state to load.
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
        """Moves the context to ``torch.cuda.current_device()``.
        """
        return self.to(f"cuda:{torch.cuda.current_device()}")

    def cpu(self):
        """Moves the context to the cpu.
        """
        return self.to("cpu")

    def ddp(self, wrap_nn_module=True, silence_recorders=True):
        """Moves the context to a distributed data-parallell setting. Can only
        be used if torch.distributed is initialized. The flags passed to
        this function allows you to toggle some behavior of the transform.

        Args:
            wrap_nn_module (bool, optional): Controls if torch.nn.Module objects should be wrapped 
                in :obj:`torch.nn.parallel.DistributedDataParallel`. Defaults to True.
            silence_recorders (bool, optional): Controls if all recorders on all non-master processes should 
                be silenced. Silencing recorders means that only the recorder in the master process is active, 
                and that all recording done in non-master processes is ignored. Defaults to True.
        """
        for name, item in vars(self).items():
            setattr(self, name, _to_distributed_data_parallel(item))
        
        return self

    def autotransform(self, wrap_nn_modules=True, load_checkpoint=True, silence_recorders=True):
        """Transforms the context according to the current experiment
        parameters. More specifically it; loads a state_dict from the parameter
        ``checkpoint`` if set, moves to cuda if paramter ``cuda`` is set, moves
        to distributed if parameter ``distributed`` is set. The flags passed to
        this function allows you to toggle some behavior of the transform.

        Args:
            wrap_nn_modules (bool, optional): Controls if torch.nn.Module objects should be wrapped 
                in :obj:`torch.nn.parallel.DistributedDataParallel`. Defaults to True.
            load_checkpoint (bool, optional): Controls the checkpoint given in the experiment parameter 
                ``checkpoint`` should be loaded (if the parameter has been set). Defaults to True.
            silence_recorders (bool, optional): Controls if all recorders on all non-master processes should 
                be silenced. Silencing recorders means that only the recorder in the master process is active, 
                and that all recording done in non-master processes is ignored. Defaults to True.
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


class SmartDataLoader(torch.utils.data.DataLoader):
    """Extends the :obj:`torch.utils.data.DataLoader` with the following:
        
        * A loading bar is displayed when iterating the dataloader.
        * The items yielded when iterating are moved to the device previously
          set with :meth:`to`.
        * Transparently handles both distributed and non-distributed modes.
    """

    def __init__(self, dataset, shuffle=False, random_seed=None, move_output=True, loading_bar=True, **kwargs):
        super().__init__(dataset, sampler=create_sampler(dataset, shuffle, random_seed), **kwargs)
        self._move_output = move_output
        self._show_loading_bar = loading_bar
        self._device = None

    def to(self, device):
        """Sets the device that yielded items should be placed on when iterating.

        Args:
            device (str, torch.Device): The device to move the items to.
        """
        self._device = device
        return self

    def __iter__(self):

        is_master = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

        if self._show_loading_bar and is_master:
            iterable = tqdm.tqdm(super().__iter__(), leave=True)

        else:
            iterable = super().__iter__()

        if self._move_output and self._device is not None:
            for item in iterable:
                yield _move_to_device(item, self._device)

        else:
            yield from iterable


def create_sampler(dataset, shuffle=True, seed=None):
    """Returns a DistributedSampler if running in distributed mode, otherwise a normal sampler

    Args:
        dataset (torch.utils.data.Dataset): The dataset the sampler will work on.
        shuffle (bool): If the sampler should be random or not.
    """

    if torch.distributed.is_initialized():
        return BetterDistributedSampler(
            dataset=dataset, 
            shuffle=shuffle,
            seed=seed
        )

    if shuffle:
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        return torch.utils.data.RandomSampler(data_source=dataset, generator=g)
    
    return torch.utils.data.SequentialSampler(data_source=dataset)


class BetterDistributedSampler(torch.utils.data.distributed.DistributedSampler):
    """This class extends torch's default DistributedSampler but removes the need
    for manually calling the set_epoch method to reseed the random sampler
    """
    def __init__(self, dataset, shuffle=True, seed=None):
        super().__init__(dataset, shuffle=shuffle, seed=seed)
        self.epoch = 0

    def __iter__(self):
        self.set_epoch(self.epoch+1)
        return super().__iter__()


class DistributedSampler(torch.utils.data.distributed.Sampler):

    def __init__(self, dataset, shuffle=True, seed=0):
        if not torch.distributed.is_initialized():
            raise Exception("Distributed sampler can only be used in a distributed environment.")

        self.dataset = dataset
        self.num_replicas = torch.distributed.get_world_size()
        self.rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.num_samples = int(math.ceil(len(self.dataset) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.random_gen = torch.Generator()
        
        if seed is not None:
            self.random_gen.manual_seed(seed)

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), self.random_gen).cuda()

            torch.distributed.broadcast(indices, 0)
            indices = indices.cpu().tolist()
        else:
            indices = list(range(len(self.dataset)))

        indices += indices[:(self.total_size - len(indices))]
        indices = indices[self.rank:self.total_size:self.num_replicas]
        
        assert len(indices) == self.num_samples, "{} != {}".format(len(indices), self.num_samples)

        return iter(indices)

    def __len__(self):
        return self.num_samples
