FROM rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1

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

# SageAttention: use native HIP WMMA backend (faster than Triton for short sequences)
ENV SAGEATTN_BACKEND=native

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
# - Two backends: 'triton' (default) and 'native' (HIP WMMA, 17-36% faster)
# - We use SAGEATTN_BACKEND=native (set above) for best perf on short sequences
#
# NOTE: This fork may fail to build against newer ROCm/PyTorch versions.
# If it fails, the image still works with flash-attention alone.
# To attempt a manual install later from inside the container:
#   cd /opt/sageattention-rdna3 && GPU_ARCHS=gfx1151 pip install . --no-build-isolation
# -------------------------------------------------------------------

RUN cd /opt && \
    git clone https://github.com/LuXuxue/sageattention-rdna3.git && \
    cd sageattention-rdna3 && \
    (GPU_ARCHS=gfx1151 pip install . --no-build-isolation 2>&1 || \
     (echo "WARNING: --no-build-isolation failed, retrying with build isolation..." && \
      GPU_ARCHS=gfx1151 pip install . 2>&1) || \
     echo "WARNING: SageAttention build failed, continuing without it. The image still has flash-attention.")

# Cloning and installing ComfyUI in case the user doesn't provide their own
# - Nothing unusual here afaik

RUN cd /opt && \
    git clone https://github.com/comfyanonymous/ComfyUI ComfyUI-pre-cloned && \
    cd ComfyUI-pre-cloned && \
    pip3 install -r requirements.txt

# Some utilities to make life/debugging easier
# Feel free to remove these if you're building from scratch locally.

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
#   --gpu-only                 Force execution on APU compute units (not CPU fallback)
#   --disable-mmap             CRITICAL: mmap > 64GB is broken on gfx1151 (ROCm bug),
#                              causes extreme slowdowns and hangs
#   --disable-smart-memory     Let ROCm/TTM handle unified memory, not ComfyUI
#   --bf16-vae                 BF16 VAE decoding prevents OOM on RDNA3.5
#   --cache-none               Disable model caching for aggressive GTT management
# -------------------------------------------------------------------

EXPOSE 8188

CMD /opt/comfyui-gfx1151-utils/check-comfyui.sh && \
    python3 /opt/ComfyUI/main.py \
        --listen 0.0.0.0 \
        --use-flash-attention \
        --gpu-only \
        --disable-mmap \
        --disable-smart-memory \
        --bf16-vae \
        --cache-none
