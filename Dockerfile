# Use an official PyTorch runtime with CUDA support
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

WORKDIR /workspace

# Install system-level dependencies required for igraph and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project configuration and source files
COPY pyproject.toml .
COPY README.md .
COPY src/ src/

# Install the package and all dependencies in editable mode
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Default command to run embedding extraction or training
CMD ["python", "run_umap.py"]
