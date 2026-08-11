FROM rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1

# -------------------------------------------------------------------
# Create a dedicated Python virtual environment
#   - --system-site-packages gives access to the base image's PyTorch/ROCm
#   - Everything else (flash-attn, sageattn, ComfyUI deps) is installed here
#   - The venv is tarred at build time and extracted at first run into the
#     host-mounted /opt/venv (see check-comfyui.sh). This way custom nodes
#     can pip install at runtime and packages survive container restarts.
# -------------------------------------------------------------------
RUN python3 -m venv /opt/venv --system-site-packages
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"

# -------------------------------------------------------------------
# Performance environment variables for Strix Halo (gfx1151 / RDNA3.5)
# -------------------------------------------------------------------

# Enable Triton-backed flash-attention on AMD (uses AMD's built-in Triton)
ENV FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE

# Enable experimental AOTriton kernels for RDNA3
ENV TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

# Prefer hipBLASLt (tuned for gfx1151) over standard hipBLAS for matmul
ENV TORCH_BLAS_PREFER_HIPBLASLT=1

# Use expandable memory segments to reduce VRAM fragmentation on unified memory
ENV PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

# SageAttention backend: "triton" (works without HIP compilation, default) or "native"
# We use "triton" since the native HIP extension uses APIs removed in ROCm 7.2
ENV SAGEATTN_BACKEND=triton

# -------------------------------------------------------------------
# "Installing" flash-attention
# - Not sure if branch `main_perf` is actually better,
#   but everyone seems to be using it, so ¯\_(ツ)_/¯
# - As far as I understood, we're not actually installing-installing
#   flash-attention, but we're telling it to use the Triton backend/implementation
#   which is already included in the rocm/pytorch image (built by amd),
#   so the lines below are executed pretty fast.
# - That ^ is what the env variable seems to be for.

RUN cd /opt && \
    git clone https://github.com/ROCm/flash-attention.git && \
    cd flash-attention && \
    git checkout main_perf && \
    python setup.py install

# -------------------------------------------------------------------
# SageAttention RDNA3 (community port for gfx11xx / Strix Halo)
# - Official SageAttention is NVIDIA-only (Ampere/Ada/Hopper/Blackwell)
# - jammm/SageAttention jam/gfx12-abi3 branch is RDNA4-only (gfx120x)
# - LuXuxue/sageattention-rdna3 is the RDNA3 fork that works on gfx1151
# - Two backends: 'triton' (default, pure Triton JIT) and 'native' (HIP WMMA)
#
# We skip the native HIP extension (SAGEATTN_SKIP_BUILD=1) because:
#   1. The HIP code uses APIs renamed in ROCm 7.2 (__hip_bfloat16, __bfloat162float)
#   2. Triton backend auto-tunes for gfx1151 without any HIP compilation
#   3. Native backend is faster for short sequences but requires ROCm 7.14+
#   To build the native backend manually:
#     docker exec comfyui-gfx1151 bash -c 'cd /opt/sageattention-rdna3 && GPU_ARCHS=gfx1151 pip install . --no-build-isolation'
# -------------------------------------------------------------------

RUN cd /opt && \
    git clone https://github.com/LuXuxue/sageattention-rdna3.git && \
    cd sageattention-rdna3 && \
    SAGEATTN_SKIP_BUILD=1 pip install . --no-build-isolation 2>&1 | tail -3 && \
    python -c "from sageattention import sageattn; print('SageAttention (Triton backend) installed successfully')"

# Cloning and installing ComfyUI in case the user doesn't provide their own
# - Nothing unusual here afaik

RUN cd /opt && \
    git clone https://github.com/comfyanonymous/ComfyUI ComfyUI-pre-cloned && \
    cd ComfyUI-pre-cloned && \
    pip install -r requirements.txt

# Some utilities to make life/debugging easier
# Feel free to remove these if you're building from scratch locally.

# Package the venv so it can be extracted into a host-mounted directory at runtime.
# If the user mounts ./venv:/opt/venv, the tarball is extracted on first run.
RUN tar czf /opt/venv-template.tar.gz -C /opt venv

RUN mkdir -p /opt/comfyui-gfx1151-utils

WORKDIR /opt/comfyui-gfx1151-utils

ADD scripts/check-comfyui.sh check-comfyui.sh
RUN chmod +x check-comfyui.sh

ADD scripts/test-pytorch.sh test-pytorch.sh
RUN chmod +x test-pytorch.sh

ADD scripts/test-pytorch-flashattention.py test-pytorch-flashattention.py
RUN chmod +x test-pytorch-flashattention.py

ADD scripts/test-sageattention.py test-sageattention.py
RUN chmod +x test-sageattention.py

# -------------------------------------------------------------------
# Run ComfyUI
#
# Flags explained:
#   --listen 0.0.0.0          Accept connections from any host
#   --use-flash-attention      Use Triton-backed flash-attention (set up above)
#   --gpu-only                 Keep everything on GPU (text encoders, VAE, CLIP).
#                              Sets HIGH_VRAM internally - models stay loaded.
#                              NOTE: --gpu-only and --highvram are mutually
#                              exclusive; --gpu-only is the stronger option.
#   --disable-mmap             CRITICAL: mmap > 64GB is broken on gfx1151 (ROCm bug),
#                              causes extreme slowdowns and hangs
#   --disable-smart-memory     Let ROCm/TTM handle unified memory, not ComfyUI
#   --bf16-vae                 BF16 VAE decoding prevents OOM on RDNA3.5
#
# Removed: --cache-none (was causing models to unload between runs)
# Removed: --highvram (mutually exclusive with --gpu-only)
# -------------------------------------------------------------------

EXPOSE 8188

CMD /opt/comfyui-gfx1151-utils/check-comfyui.sh && \
    export PATH="/opt/venv/bin:$PATH" && \
    python /opt/ComfyUI/main.py \
        --listen 0.0.0.0 \
        --use-flash-attention \
        --gpu-only \
        --disable-mmap \
        --disable-smart-memory \
        --bf16-vae
