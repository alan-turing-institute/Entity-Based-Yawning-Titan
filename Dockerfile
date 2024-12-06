FROM python:3.9.20

ENV PYTHONUNBUFFERED 1
ENV PIP_NO_CACHE_DIR 1
ENV PIP_ROOT_USER_ACTION=ignore

RUN pip install setuptools==66 wheel==0.38.4
RUN pip install gym==0.21.0
RUN pip install typing_extensions hyperstate enn_trainer
RUN pip install torch==1.12.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-index torch-scatter -f https://data.pyg.org/whl/torch-1.12.0+cpu.html

RUN pip install stable_baselines3==1.6.2 wandb platformdirs networkx tinydb tabulate
COPY yawning_titan /app/yawning_titan
COPY src /app/src
COPY README.md /app
COPY pyproject.toml /app
RUN pip install -e /app/yawning_titan
RUN pip install -e /app
WORKDIR /app

CMD /bin/bash