FROM python:3.10-slim-bullseye AS base

# Get system deps
#RUN apt-get update && \
#    apt-get install --no-install-recommends -y python3-opencv python3-numpy llvm && \
#    rm -rf /var/lib/apt/lists/*

# Get requirements out of the way first
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir llvmlite>=0.37.0 && \
    pip install --no-cache-dir -r requirements.txt

# Get su-exec
#RUN curl -o /usr/local/bin/su-exec.c https://raw.githubusercontent.com/ncopa/su-exec/master/su-exec.c
#RUN gcc -Wall /usr/local/bin/su-exec.c -o/usr/local/bin/su-exec
#RUN chown root:root /usr/local/bin/su-exec
#RUN chmod 0755 /usr/local/bin/su-exec

# Copy the rest of the code
COPY . /opt/oplab_pipeline
WORKDIR /opt/oplab_pipeline

# Install the code
RUN pip install -U --no-cache-dir .

# Set up the entrypoint
RUN mkdir -p /data
WORKDIR /data
#ENTRYPOINT ["/home/oplab_docker/oplab_pipeline/docker_entrypoint.sh"]
CMD ["/bin/bash"]
