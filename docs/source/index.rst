.. pystematic-torch documentation master file, created by
   sphinx-quickstart on Tue Aug 17 18:21:15 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Introduction
============

This is an extension to pystematic that adds functionality related to running
machine learning experiments in pytorch. Its main contribution is the
:class:`pystematic.torch.Context` and related classes.

Installation
============

All you have to do for pystematic to load the extension is to install it:

.. code-block:: 

    $ pip install pystematic-torch


Experiment API
==============

This plugin adds some API functionality under the ``pystematic.torch``
namespace.

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
:meth:`Context.state_dict` and :meth:`Context.load_state_dict` methods.

Here is a short example showing how the Context may be used:

.. code-block:: python

    import pystematic

    ctx = pystematic.torch.TorchContext()

    ctx.model = torch.nn.Sequential(
        torch.nn.Linear(2, 1),
        torch.nn.Sigmoid()
    )
    
    # We use the smart dataloader so that batches are moved to the correct
    # device
    ctx.dataloader = pystematic.torch.SmartDataLoader(
        dataset=Dataset(),
        batch_size=2
    )
    ctx.loss_function = torch.nn.BCELoss()

    ctx.cuda() # Move everything to cuda

    # Remember to initialize the optimizer after moving
    ctx.optimzer = torch.optim.SGD(ctx.model.parameters(), lr=0.01)

    for input, lbl in ctx.dataloader:

        output = ctx.model(input)
        
        loss = ctx.loss_function(output, lbl)

        ctx.optimzer.zero_grad()
        loss.backward()
        ctx.optimzer.step()


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
