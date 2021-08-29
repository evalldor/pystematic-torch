This is an extension to `pystematic <https://github.com/evalldor/pystematic>`_
that adds functionality related to running machine learning experiments in
pytorch. Its main contribution is the ``Context`` object and related classes.
Which provides an easy way to manage all pytorch related objects.


Installation
============

All you have to do for pystematic to find the plugin is to install it:

.. code-block:: 

    $ pip install pystematic-torch


Example
=======

Here's a small example that shows how using the ``Context`` object,
``SmartDataLoader`` and ``Recorder`` simplifies setting up and running a
training session in pytorch.

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


Documentation
=============

Reference documentation is available at
`<https://pystematic-torch.readthedocs.io>`_.
