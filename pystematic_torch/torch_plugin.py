import logging
import pathlib

import pystematic as ps
import pystematic.core as core
import torch

from . import context, recording, utils


logger = logging.getLogger('pystematic.torch')


class TorchPlugin:

    def __init__(self, app) -> None:
        self.api_object = TorchApi()

        app.on_experiment_created(self.experiment_created)
        app.on_before_experiment(self.api_object._before_experiment)
       
        self.extend_api(app.get_api_object())
    
    def experiment_created(self, experiment):
        """Gives the plugin a chance to modify an experiment when it is created
        """
        for param in pytorch_params:
            experiment.add_parameter(param)
        
        return experiment

    def extend_api(self, api_object):
        setattr(api_object, "torch", self.api_object)


class TorchApi:

    def _before_experiment(self, experiment, params):
        
        if params["distributed"]:
            self.init_distributed()

    def move_to_device(self, device, *args):
        """Utility method to place a batch of data on a specific device (i.e.
        cuda or cpu). It handles nested dicts and lists by traversing every
        element and moving them to the proper device if possible. Unrecognized
        objects will be left as is.

        Args:
            device (str, torch.Device): The device to move to
            *args (any): Any objects that you want to move

        Returns:
            any: The moved objects
        """
        
        if len(args) == 1:
            return context._move_to_device(args[0], device)
        
        return context._move_to_device(args, device)

    def save_checkpoint(self, state_dict, id) -> None:
        """Saves the provided state_dict to a file in ``pystematic.output_dir``.
        This function will make sure to only save the checkpoint in the master
        process when called in distributed mode.
        
        Args:
            state_dict (dict): The state dict to save, such as the on returned from 
                :meth:`Context.state_dict`
            id (any): An id that uniquely identifies this checkpoint. E.g. epoch number, 
                step number etc.
        """

        if self.is_master():
            checkpoint_file_path = ps.output_dir.joinpath(f"checkpoint-{id}.pt")

            logger.info(f"Saving checkpoint '{checkpoint_file_path}'.")

            with checkpoint_file_path.open("wb") as f:
                torch.save(state_dict, f)

    def load_checkpoint(self, checkpoint_file_path) -> dict:
        """Loads and returns a checkpoint from the given filepath.

        Args:
            checkpoint_file_path (str, pathlib.Path): Path to the file to load.

        Returns:
            dict: The loaded state dict.
        """
        with open(checkpoint_file_path, "rb") as f:
            return torch.load(f, map_location="cpu")

    def run_parameter_sweep(self, experiment, list_of_params, max_num_processes=1, num_gpus_per_process=None) -> None:
        """Extends the :func:`pystematic.run_parameter_sweep` with GPU limiting
        capabilities.

        Runs an experiment multiple times with a set of different params. At
        most :obj:`max_num_processes` concurrent processes will be used. This
        call will block until all experiments have been run.

        Args:
            experiment (Experiment): The experiment to run.
            list_of_params (list of dict): A list of parameter dictionaries. Each corresponding to 
                one run of the experiment. See :func:`pystematic.param_matrix` for a convenient way 
                of generating such a list.
            max_num_processes (int, optional): The maximum number of concurrent processes to use 
                for running the experiments. Defaults to 1.
            num_gpus_per_process (int, optional): The number of GPUs to allocate for each experiment. 
                If None no allocation is done. Default is None.
        """

        pool = utils.ProcessQueue(max_num_processes, range(torch.cuda.device_count()), num_gpus_per_process)
        pool.run_and_wait_for_completion(experiment, list_of_params)

    #
    # Pytorch distributed data parallel
    #

    def init_distributed(self) -> None:
        """Initializes a distributed runtime. This function is called automatically 
        during initialization if the parameter ``distributed`` is set to ``True``.
        """
        if not ps.is_subprocess():
            for i in range(ps.params["nproc_per_node"]-1):
                ps.launch_subprocess()


        global_rank = ps.params["nproc_per_node"] * ps.params["node_rank"] + ps.params["local_rank"]
        world_size = ps.params["nproc_per_node"] * ps.params["nnodes"]

        logger.debug(f"Initializing distributed runtime (world size '{world_size}', "
                    f"local rank '{ps.params['local_rank']}', global rank '{global_rank}')...")

        torch.cuda.set_device(ps.params["local_rank"])

        torch.distributed.init_process_group(
            backend='nccl',
            init_method=f"tcp://{ps.params['master_addr']}:{ps.params['master_port']}",
            world_size=world_size,
            rank=global_rank
        )

        logger.debug(f"Distributed runtime initialized.")

    def is_distributed(self) -> bool:
        """Alias for ``torch.distributed.is_initialized()``. 

        Returns:
            bool: Returns true if torch distributed runtime is initialized.
        """
        return torch.distributed.is_initialized()

    def is_master(self) -> bool:
        """If running in distributed mode, returns whether of not this current
        process is the master process. In non-distributed mode, always returns True.

        Returns:
            bool: Whether the current process is the master process.
        """
        return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

    def get_num_processes(self) -> int:
        """Alias for ``torch.distributed.get_world_size()``. In non-distributed
        mode, this always returns 1.

        Returns:
            int: The total number of processes in the distributed runtime.
        """
        if torch.distributed.is_initialized():
            return torch.distributed.get_world_size()

        return 1

    def get_rank(self) -> int:
        """Returns the global rank of the current process. If the current
        process is not currently running in distributed mode, it always return 0.
        In single node training the rank is the same as the local rank.

        Returns:
            int: The rank of the current process.
        """
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank()

        return 0

    def broadcast_from_master(self, value):
        """Alias for ``torch.distributed.broadcast(value, 0)``. In
        non-distributed mode, this just returns the value.
        """
        value = torch.tensor(value)

        if torch.distributed.is_initialized():
            torch.distributed.broadcast(value, 0)

        return value

    def distributed_barrier(self) -> None:
        """Alias for ``torch.distributed.barrier()``. In non-distributed mode,
        this is a noop.
        """
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    Context = context.Context
    SmartDataLoader = context.SmartDataLoader
    Recorder = recording.Recorder


pytorch_params = [
    core.Parameter(
        name="checkpoint",
        type=pathlib.Path,
        help="If using the context autotransform() method, it will "
        "load the checkpoint pointed to by this parameter (if set)."
    ),
    core.Parameter(
        name="cuda",
        default=True,
        is_flag=True,
        help="If using the context autotransform() method, "
        "setting this to True will move the context to cuda."
    ),
    core.Parameter(
        name="distributed",
        default=False,
        is_flag=True,
        help="Controls if the experiment should be run in a distributed "
        "fashion (multiple GPUs). When set to True, a distributed mode "
        "will be launched (similar to torch.distributed.launch) "
        "before the experiment main function is run. If using the context "
        "autotransform() method, this parameter also tells the context "
        "whether to move to distributed mode (ddp)."
    ),
    core.Parameter(
        name="nproc_per_node",
        envvar="NPROC_PER_NODE", 
        type=int, 
        default=1,
        help="The number of processes to launch on each node, for GPU "
        "training, this is recommended to be set to the number of GPUs "
        "in your system so that each process can be bound to a single GPU."
    ),
    core.Parameter(
        name="node_rank", 
        envvar="NODE_RANK",
        type=int, 
        default=0,
        help="The rank of the node for multi-node distributed training."
    ),
    core.Parameter(
        name="nnodes", 
        envvar="NNODES",
        type=int, 
        default=1,
        help="The number of nodes to use for distributed training."
    ),
    core.Parameter(
        name="master_addr", 
        default="127.0.0.1",
        envvar="MASTER_ADDR",
        type=str,
        help="The master node's (rank 0) IP address or the hostname. "
        "Leave default for single node training."
    ),
    core.Parameter(
        name="master_port", 
        default=29500, 
        envvar="MASTER_PORT",
        type=int,
        help="The master node's (rank 0) port used for communciation "
        "during distributed training."
    )
]

