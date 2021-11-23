FROM python:3.8-slim  AS base

# first stage
FROM base AS builder
COPY requirements.txt .
# install dependencies to the local user directory (eg. /root/.local)
RUN pip install -U pip && \
    pip install --user -r requirements.txt && \
    rm -rf /root/.cache/pip

# second stage
FROM base AS release
# Install OpenCV depencencies
RUN apt-get update && apt-get install -y python3-opencv
COPY --from=builder /root/.local /root/.local
WORKDIR /code
COPY . .
ENV PATH=/root/.local/bin:$PATH
RUN pip install --user . && \
    rm -rf /root/.cache/pip

# Make RUN commands use the new environment:
CMD ["/bin/bash"]

