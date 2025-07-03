# Use the official PyTorch image with CUDA 11.8
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies including gcc
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install boltz
RUN pip install --upgrade pip && \
    pip install boltz -U

# Clone the ProteinMPNN repository
RUN git clone https://github.com/dauparas/ProteinMPNN.git /opt/ProteinMPNN

# Add ProteinMPNN to PATH so scripts are accessible anywhere
ENV PATH="/opt/ProteinMPNN:${PATH}"

# Default command
CMD ["bash"]

