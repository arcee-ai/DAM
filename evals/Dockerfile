
# Use the specified base image
FROM runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04

ARG HUGGINGFACE_TOKEN

# Install necessary packages and tools
RUN apt update && \
    apt install -y screen vim git-lfs && \
    pip install huggingface_hub[cli]

COPY lm-evaluation-harness /lm-evaluation-harness
COPY post_start.sh /post_start.sh

WORKDIR /lm-evaluation-harness

RUN pip install -e ".[ja]"


# Optional: Set the working directory
# WORKDIR 

# Optional: Add any entrypoint or command if required
CMD ["/start.sh"]