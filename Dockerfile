# Build me with
# docker build --tag oceanperception/oplab_pipeline -f Dockerfile .

# Run me with the below line
# Replace HOST_FOLDER with the path to the folder that you want to use in the Docker session. In the Docker session that location will be mapped to /data
# docker run -e USER=$(whoami) -h $HOSTNAME --user $(id -u):$(id -g) -v /etc/passwd:/etc/passwd:ro --rm -it --ipc=private --name=oplab_pipeline_$(whoami)_$(date +%Y%m%d_%H%M%S) -v HOST_FOLDER:/data oceanperception/oplab_pipeline


FROM python:3.11-slim-bookworm AS base

# Get requirements out of the way first
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --extra-index-url https://rospypi.github.io/simple/ rosbag roslz4 && \
    pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy the rest of the code
COPY . /opt/oplab_pipeline
WORKDIR /opt/oplab_pipeline

# Install the code
RUN pip install -U --no-cache-dir .

# Set up the entrypoint
RUN mkdir -p /data
WORKDIR /data
CMD ["/bin/bash"]
