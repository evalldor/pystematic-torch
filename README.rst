A `pystematic <https://github.com/evalldor/pystematic>`_ plugin for running
experiments in pytorch. 

This is an extension to pystematic that adds functionality related to running
machine learning experiments in pytorch. Its main contribution is the
``ContextObject`` and related classes. Which provides an easy way to manage all
pytorch related objects.

Documentation is in the works.

Quickstart
==========

Installation
------------

All you have to do for pystematic to find the plugin is to install it:

.. code-block:: 

    $ pip install pystematic-torch


Context objects
---------------

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
``context.cpu()``, ``context.cuda()`` and ``context.ddp()`` methods. You can
also serialize and restore the state of the entire session with the
``context.state_dict()`` and ``context.load_state_dict()`` methods.
