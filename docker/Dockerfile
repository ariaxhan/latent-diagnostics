# Injection Detection Server
# Model loads once at startup, stays in memory

FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy source
COPY src/ ./src/
COPY docker/server.py .

# Pre-download models at build time (cached in image)
RUN python -c "from transformer_lens import HookedTransformer; HookedTransformer.from_pretrained('gemma-2-2b', device='cpu')" || true
RUN python -c "from sae_lens import SAE; SAE.from_pretrained(release='gemma-scope-2b-pt-res-canonical', sae_id='layer_5/width_16k/canonical', device='cpu')" || true
RUN python -c "from sae_lens import SAE; SAE.from_pretrained(release='gemma-scope-2b-pt-transcoders', sae_id='layer_5/width_16k/average_l0_153', device='cpu')" || true

# Expose port
EXPOSE 8000

# Start server (model loads once)
CMD ["python", "server.py"]
