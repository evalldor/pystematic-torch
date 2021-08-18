.. pystematic-torch documentation master file, created by
   sphinx-quickstart on Tue Aug 17 18:21:15 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pystematic-torch's documentation!
============================================

This is an extension to pystematic that adds functionality related to running
machine learning experiments in pytorch. Its main contribution is the
:class:`pystematic.torch.ContextObject` and related classes.


Quickstart
==========

Installation
------------

All you have to do for pystematic to load the extension is to install it:

.. code-block:: 

    $ pip install pystematic-torch


Context objects
---------------

When you are developing a model in pytorch, you often want to be able to train
the model in many different settings, such as distributed, single gpu or even
just on the cpu depending on your work location and on available resources. The
main purpose of the context object is to allow you to transition seamlessly
between these different modes of training, without changing your code. 



.. toctree::
   :maxdepth: 1

   apidoc