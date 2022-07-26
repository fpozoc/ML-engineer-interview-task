FROM python:3.10
RUN apt-get update && apt-get install -y python-dev libffi-dev build-essential
WORKDIR /workspace
COPY ./setup.py /workspace
RUN pip install .
RUN pip install jupyterlab
EXPOSE 8888
ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]