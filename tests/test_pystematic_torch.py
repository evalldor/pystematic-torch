import pytest
import pystematic

import torch


def test_main_function_is_run():

    class CustomException(Exception):
        pass

    @pystematic.experiment
    def exp(params):
        assert "local_rank" in params
        raise CustomException()

    with pytest.raises(CustomException):
        exp.cli([])

    with pytest.raises(CustomException):
        exp.run({})


def test_context_cpu():
    ctx = pystematic.torch.Context()

    ctx.model = torch.nn.Sequential(
        torch.nn.Linear(2, 1),
        torch.nn.Sigmoid()
    )
    ctx.dataloader = pystematic.torch.SmartDataLoader(
        dataset=Dataset(),
        batch_size=2
    )
    
    ctx.loss_function = torch.nn.BCELoss()
    ctx.optimzer = torch.optim.SGD(ctx.model.parameters(), lr=0.01)
    
    with pytest.raises(Exception):
        ctx.cpu() # Having an optimizer present when moving

    
    for input, lbl in ctx.dataloader:

        output = ctx.model(input)
        
        loss = ctx.loss_function(output, lbl)

        ctx.optimzer.zero_grad()
        loss.backward()
        ctx.optimzer.step()


def test_move_context_to_cuda():
    ctx = pystematic.torch.Context()

    ctx.model = torch.nn.Sequential(
        torch.nn.Linear(2, 1),
        torch.nn.Sigmoid()
    )
    ctx.dataloader = pystematic.torch.SmartDataLoader(
        dataset=Dataset(),
        batch_size=2
    )
    ctx.loss_function = torch.nn.BCELoss()

    ctx.cuda()
    ctx.optimzer = torch.optim.SGD(ctx.model.parameters(), lr=0.01)


    for input, lbl in ctx.dataloader:

        output = ctx.model(input)
        
        loss = ctx.loss_function(output, lbl)

        ctx.optimzer.zero_grad()
        loss.backward()
        ctx.optimzer.step()


def test_smart_dataloader():
    ctx = pystematic.torch.Context()

    ctx.model = torch.nn.Sequential(
        torch.nn.Linear(2, 1),
        torch.nn.Sigmoid()
    )
    ctx.dataloader = pystematic.torch.SmartDataLoader(
        dataset=Dataset(),
        batch_size=2,
        move_output=False,
        loading_bar=False
    )
    ctx.loss_function = torch.nn.BCELoss()

    ctx.cuda()
    ctx.optimzer = torch.optim.SGD(ctx.model.parameters(), lr=0.01)

    with pytest.raises(Exception):
        for input, lbl in ctx.dataloader:
            
            output = ctx.model(input)
            
            loss = ctx.loss_function(output, lbl)

            ctx.optimzer.zero_grad()
            loss.backward()
            ctx.optimzer.step()


    ctx = pystematic.torch.Context()

    ctx.model = torch.nn.Sequential(
        torch.nn.Linear(2, 1),
        torch.nn.Sigmoid()
    )
    ctx.dataloader = pystematic.torch.SmartDataLoader(
        dataset=Dataset(),
        batch_size=2,
        loading_bar=False
    )
    ctx.loss_function = torch.nn.BCELoss()

    ctx.optimzer = torch.optim.SGD(ctx.model.parameters(), lr=0.01)

    num_iterations = 0

    for input, lbl in ctx.dataloader:
        num_iterations += 1
        output = ctx.model(input)
        
        loss = ctx.loss_function(output, lbl)

        ctx.optimzer.zero_grad()
        loss.backward()
        ctx.optimzer.step()

    assert num_iterations == 3


def test_move_context_to_ddp():
    ddp_exp.run({
        "distributed": True,
        "nproc_per_node": 2
    })


def test_context_state_dict():
    pass


class Dataset:
        
    def __len__(self):
        return 6

    def __getitem__(self, index):
        data = torch.randn(2)
        lbl = torch.randint(0, 2, (1,)).float()
        
        return data, lbl


@pystematic.experiment
def ddp_exp(params):
    ctx = pystematic.torch.Context()

    ctx.model = torch.nn.Sequential(
        torch.nn.Linear(2, 1),
        torch.nn.Sigmoid()
    )
    ctx.dataloader = pystematic.torch.SmartDataLoader(
        dataset=Dataset(),
        batch_size=2
    )
    ctx.loss_function = torch.nn.BCELoss()

    ctx.cuda()
    ctx.ddp()

    ctx.optimzer = torch.optim.SGD(ctx.model.parameters(), lr=0.01)


    for input, lbl in ctx.dataloader:

        output = ctx.model(input)
        
        loss = ctx.loss_function(output, lbl)

        ctx.optimzer.zero_grad()
        loss.backward()
        ctx.optimzer.step()

