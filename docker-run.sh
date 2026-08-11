#!/bin/bash

docker run -it \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --group-add render \
  --ipc=host \
  -e FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
  -e TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
  -e TORCH_BLAS_PREFER_HIPBLASLT=1 \
  -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
  -e SAGEATTN_BACKEND=triton \
  -p 8188:8188 \
  -v $(pwd)/ComfyUI:/opt/ComfyUI \
  -v $(pwd)/venv:/opt/venv \
  --shm-size 8G \
  --name comfyui-gfx1151 \
  arczewski/comfyui-gfx1151:latest
