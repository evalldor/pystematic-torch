API reference
=============

Experiment API
--------------

This plugin adds some API functionality under the ``pystematic.torch``
namespace.

General
+++++++

.. autofunction:: pystematic.torch.place_on_correct_device

.. autofunction:: pystematic.torch.iterate

.. autofunction:: pystematic.torch.save_checkpoint

.. autofunction:: pystematic.torch.load_checkpoint

.. autofunction:: pystematic.torch.run_parameter_sweep


Distributed
+++++++++++

.. autofunction:: pystematic.torch.init_distributed

.. autofunction:: pystematic.torch.is_distributed

.. autofunction:: pystematic.torch.is_master

.. autofunction:: pystematic.torch.get_num_processes

.. autofunction:: pystematic.torch.get_rank

.. autofunction:: pystematic.torch.broadcast_from_master

.. autofunction:: pystematic.torch.distributed_barrier

Context
+++++++

A context object is like a big container that holds all pytorch related
objects you need. Its main use is to allow a pytorch session to transition
seamlessly between different modes (e.g. distributed, cuda) based on
experiment parameters. It does this by transparently transforming some
objects that you add. For example, when running in distributed mode, all
pytorch models added to this context will be automatically wrapped in 
torch's :obj:`DistributedDataParallel`.

The methods :meth:`state_dict` and :meth:`load_state_dict` provides a single
point of entry to the state of the entire session (provided that all objects
are registered with it).

.. autoclass:: pystematic.torch.ContextObject
    :members: has, cuda, cpu, ddp, state_dict, load_state_dict, autotransform
    :undoc-members:

.. autoclass:: pystematic.torch.ContextDict
    :members: items, keys, values, cuda, cpu, ddp, state_dict, load_state_dict, autotransform
    :undoc-members:

.. autoclass:: pystematic.torch.ContextList
    :members: insert, append, cuda, cpu, ddp, state_dict, load_state_dict, autotransform
    :undoc-members:

Other
+++++

.. autoclass:: pystematic.torch.Recorder

.. autoclass:: pystematic.torch.BetterDataLoader


Default parameters
------------------

The following parameters are added to all experiments by default. Note that
these are also listed if you run an experiment from the command line with the
``--help`` option.

* ``checkpoint``: Load context from checkpoint. Default value is ``None``.

* ``cuda``: Whether to run on cuda by default. Default value is ``True``.

* ``distributed``: Controls if the experiment should be run in a distributed
  fashion (multiple GPUs). Default value is ``False``.

* ``node_rank``: The rank of the node for multi-node distributed training.
  Default value is ``0``.

* ``nproc_per_node``: The number of processes to launch on each node, for GPU
  training, this is recommended to be set to the number of GPUs in your system
  so that each process can be bound to a single GPU. Default value is ``1``.

* ``nnodes``: The number of nodes to use for distributed training. Default value
  is ``1``.

* ``master_addr``: Master node (rank 0)'s address, should be either 
  the IP address or the hostname of node 0. Leave default for single node
  training. Default value is ``127.0.0.1``.

* ``master_port``: Master node (rank 0)'s free port that needs to be used for
  communciation during distributed training. Default value is ``29500``.
