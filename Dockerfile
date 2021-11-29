FROM python:3.8 AS base

# first stage
FROM base AS builder
COPY requirements.txt .
# install dependencies to the local user directory (eg. /root/.local)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# second stage
FROM base AS release
RUN apt-get update && apt-get install -y python3-opencv llvm
RUN useradd --create-home --shell /bin/bash oplab
COPY --from=builder /usr/local/lib/python3.8/site-packages /usr/local/lib/python3.8/site-packages
COPY . /home/oplab/oplab_pipeline
RUN chown -R oplab:oplab /home/oplab
USER oplab
WORKDIR /home/oplab/oplab_pipeline
ENV PATH=/home/oplab/.local/bin:$PATH
ENV PYTHONPATH=/home/oplab/.local/lib/python3.8/site-packages
RUN pip install --no-cache-dir --user .

# Make RUN commands use the new environment:
CMD ["/bin/bash"]

