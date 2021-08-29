.. pystematic-torch documentation master file, created by
   sphinx-quickstart on Tue Aug 17 18:21:15 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Introduction
============

This is an extension to pystematic that adds functionality related to running
machine learning experiments in pytorch. Its main contribution is the
:ref:`Context<index:context>` and related classes, which has the goal of making
your code agnostic to whether or not you are running on cuda, cpu, or
distributed data-parallel.


Installation
============

All you have to do for pystematic to load the extension is to install it:

.. code-block:: 

    $ pip install pystematic-torch


Experiment API
==============

This extension publishes its API under the ``pystematic.torch`` namespace.


General
-------

.. autofunction:: pystematic.torch.move_to_device

.. autofunction:: pystematic.torch.save_checkpoint

.. autofunction:: pystematic.torch.load_checkpoint

.. autofunction:: pystematic.torch.run_parameter_sweep


Distributed
-----------

.. autofunction:: pystematic.torch.is_distributed

.. autofunction:: pystematic.torch.is_master

.. autofunction:: pystematic.torch.get_num_processes

.. autofunction:: pystematic.torch.get_rank

.. autofunction:: pystematic.torch.broadcast_from_master

.. autofunction:: pystematic.torch.distributed_barrier


Context
-------

When you are developing a model in pytorch, you often want to be able to train
the model in many different settings, such as multi-node distributed, single gpu
or even just on the cpu depending on your work location and on available
resources. The main purpose of the context object is to allow you to transition
seamlessly between these different modes of training, without changing your
code. 

If you are familiar with the ``torch.nn.Module`` object, you know that whenever
you add a paramater to the object, it gets registered with it, and when you want
to move the model to another device, you simply call ``module.cuda()`` or
``module.cpu()`` to move all paramters registered with the module.

A context object is like a torch module on steroids. You are meant to register
every object important to your training session with it, e.g. models,
optimizers, epoch counter etc. You can then transition your session with the
:meth:`Context.cpu`, :meth:`Context.cuda` and :meth:`Context.ddp` methods. 

You can also serialize and restore the state of the entire session with the
:meth:`Context.state_dict` and :meth:`Context.load_state_dict` methods, which
makes checkpointing painless.

Here is a short example showing how the Context may be used:

.. code-block:: python

    import pystematic

    @pystematic.experiment
    def context_example(params):
        ctx = pystematic.torch.Context()
        ctx.epoch = 0
        ctx.recorder = pystematic.torch.Recorder()

        ctx.model = torch.nn.Sequential(
            torch.nn.Linear(2, 1),
            torch.nn.Sigmoid()
        )
        
        # We use the smart dataloader so that batches are moved to 
        # the correct device
        ctx.dataloader = pystematic.torch.SmartDataLoader(
            dataset=Dataset(),
            batch_size=2
        )
        ctx.loss_function = torch.nn.BCELoss()

        ctx.cuda() # Move everything to cuda 
        # ctx.ddp() # and maybe distributed data-parallel?

        # Remember to initialize the optimizer after moving
        ctx.optimzer = torch.optim.SGD(ctx.model.parameters(), lr=0.01)

        if params["checkpoint"]:
            # Load checkpoint
            ctx.load_state_dict(params["checkpoint"])

        # Train one epoch
        for input, lbl in ctx.dataloader:
            # The smart dataloader makes sure the batch is placed on 
            # the correct device.
            output = ctx.model(input)
            
            loss = ctx.loss_function(output, lbl)

            ctx.optimzer.zero_grad()
            loss.backward()
            ctx.optimzer.step()

            ctx.recorder.scalar("train/loss", loss)
            ctx.recorder.step()
        
        ctx.epoch += 1

        # Save checkpoint
        pystematic.torch.save_checkpoint(ctx.state_dict(), id=ctx.epoch)


The following list specifies the transformations applied to each type of
object:

:obj:`torch.nn.Module`: 

* cuda: moved to ``torch.cuda.current_device()``
* cpu: moved to cpu
* ddp: Gets wrapped in ``torch.nn.parallel.DistributedDataParallel`` and
  then in an object proxy, that delegates all non-existing ``getattr()`` calls
  to the underlying module. This means that you should be able to use any custom
  attributes and methods of the original module, even after it get wrapped in
  the DDP module. This is needed to make the code you write agnostic to whether
  or not it is currently run in distributed mode.

:obj:`torch.optim.Optimizer`:

* cuda, cpu, ddp: If an optimizer instance is encounterd any of these call
  will raise an exception. The reason is that optimizers needs to be initialized
  *after* the parameters have been placed on the correct. This is an unfortunate
  quirk of pytorch and will hopefully be fixed in the future.

:obj:`pystematic.torch.Recorder`:

* ddp: gets silenced on non master processes

:obj:`pystematic.torch.SmartDataLoader`:

* cuda, cpu: Moves the dataloader to the proper device. If you initialize
  the dataloader with ``move_output = True``, the items yielded when
  iterating the dataloader are moved to the correct device.

Any object with a method named ``to()`` (such as :obj:`torch.Tensor`):

* cuda, cpu, ddp: call the ``to()`` method with the device to move the
  object to.

All other types of objects are left unchanged.

The :meth:`autotransform` method uses the parameters ``cuda``,
``distributed``, ``checkpoint`` to automatically determine how the context
should be transformed. 

.. autoclass:: pystematic.torch.Context
    :members: cuda, cpu, ddp, state_dict, load_state_dict, autotransform
    :undoc-members:


Other
-----

.. autoclass:: pystematic.torch.Recorder
    :members: count, step, params, scalar, figure, image, state_dict, load_state_dict
    :undoc-members:

.. autoclass:: pystematic.torch.SmartDataLoader
    :members: to
    :undoc-members:


Default parameters
==================

The following parameters are added to all experiments by default. Note that
these are also listed if you run an experiment from the command line with the
``--help`` option.

* ``checkpoint``: If using the context :meth:`autotransform` method, it will
  load the checkpoint pointed to by this parameter (if set). Default value is
  ``None``.

* ``cuda``: If using the context :meth:`autotransform` method, setting this to
  True will move the context to cuda. Default value is ``True``.

* ``distributed``: Controls if the experiment should be run in a distributed
  fashion (multiple GPUs). When set to True, a distributed mode will be launched
  (similar to ``torch.distributed.launch``) before the experiment main function
  is run. If using the context :meth:`autotransform` method, this parameter also
  tells the context whether to move to distributed mode (``ddp``). Default value
  is ``False``.

* ``node_rank``: The rank of the node for multi-node distributed training.
  Default value is ``0``.

* ``nproc_per_node``: The number of processes to launch on each node, for GPU
  training, this is recommended to be set to the number of GPUs in your system
  so that each process can be bound to a single GPU. Default value is ``1``.

* ``nnodes``: The number of nodes to use for distributed training. Default value
  is ``1``.

* ``master_addr``: The master node's (rank 0) IP address or the hostname.
  Leave default for single node training. Default value is ``127.0.0.1``.

* ``master_port``: The master node's (rank 0) port used for communciation 
  during distributed training. Default value is ``29500``.
