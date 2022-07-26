FROM python:3.9
RUN apt-get update && apt-get install -y python-dev libffi-dev build-essential
WORKDIR /app
COPY . /app
ENV PYTHONPATH=${PYTHONPATH}:${PWD}
RUN pip install ".[extra]"
ENTRYPOINT ["make", "run-pipeline"]