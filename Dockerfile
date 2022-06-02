from pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

WORKDIR /pystematic-torch/

COPY / /pystematic-torch/

RUN pip install poetry

RUN rm /pystematic-torch/poetry.lock

RUN poetry install

ENTRYPOINT ["poetry", "run", "pytest", "-s"]