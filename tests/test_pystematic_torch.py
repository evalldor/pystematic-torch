import pytest
import pystematic


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


def test_move_context_from_cpu_to_cuda():
    pass


def test_move_context_from_cuda_to_ddp():
    pass


def test_context_state_dict():
    pass
