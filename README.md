# ComfyUI for gfx1151 (Ryzen AI MAX)

Dockerized ComfyUI with PyTorch, flash-attention **and SageAttention RDNA3** for gfx1151
(AMD Strix Halo, Ryzen AI Max+ 395), relying on AMD's pre-built and pre-configured
environment (no custom wheels).

Versions used:
* ROCm: 7.2
* PyTorch: 2.9.1
* Python: 3.12
* ComfyUI (built-in): v0.15.0
* SageAttention: RDNA3 fork ([LuXuxue/sageattention-rdna3](https://github.com/LuXuxue/sageattention-rdna3))
* Flash-Attention: ROCm fork, `main_perf` branch (Triton backend)

**Last updated & tested**: Feb 25, 2026, on 6.18.9 (ArchLinux), AMD RYZEN AI MAX+ 395 (Framework Desktop), 
with [opencl-amd](https://aur.archlinux.org/packages/opencl-amd) packages (7.2.0-1).

> [!NOTE]
> **SageAttention added!** This image now includes the community RDNA3 port of SageAttention
> (the official SageAttention is NVIDIA-only). The `jammm/SageAttention` `gfx12-abi3` branch
> is RDNA4-only and will NOT work on Strix Halo. We use `LuXuxue/sageattention-rdna3` compiled
> for gfx1151 with the native HIP WMMA backend for 17-36% faster attention.

> [!CAUTION]
> I kinda understand what's going on here, but not fully. It took me most of the day to figure out how to run
> ComfyUI on my Framework Desktop without it crashing (which is absurd for a CPU with _AI_ in the name), and
> the final working solution turned out to be much simpler than what I was able to find initially.
>
> That being said, it works (as of Feb 25, 2026), but I'm not sure if there's an even better / more correct way of
> achieving the same thing. Same goes for the environment variables that are supposedly making ComfyUI
> faster / resource efficient.
>
> I just want to share this solution to save someone else a couple of hours ¯\_(ツ)_/¯

## Performance optimizations

This image includes several optimizations tuned for Strix Halo (gfx1151 / RDNA3.5):

| Optimization | What it does |
|---|---|
| **SageAttention RDNA3** | Community port using HIP native WMMA kernels. 17-36% faster attention vs flash-attention on short sequences. Set via `SAGEATTN_BACKEND=native` (or `triton`) |
| **Flash-Attention (Triton)** | Uses AMD's built-in Triton backend (not compiled from source). Enabled via `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` |
| **AOTriton** | Experimental Triton kernels for RDNA3. Enabled via `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` |
| **hipBLASLt** | gfx1151-tuned matrix multiplication. Enabled via `TORCH_BLAS_PREFER_HIPBLASLT=1` |
| **Expandable segments** | Reduces VRAM fragmentation on unified memory. `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` |
| **`--disable-mmap`** | Critical! mmap > 64GB is broken on gfx1151 (ROCm bug), causes extreme slowdowns |
| **`--bf16-vae`** | BF16 VAE decoding prevents OOM on RDNA3.5 |
| **`--disable-smart-memory`** | Let ROCm/TTM handle unified memory, not ComfyUI |
| **`--cache-none`** | Disable model caching for aggressive GTT management |

### About SageAttention on AMD

Official SageAttention supports **NVIDIA only** (Ampere, Ada, Hopper, Blackwell). On AMD:
- **RDNA3 (gfx11xx)**: Use [LuXuxue/sageattention-rdna3](https://github.com/LuXuxue/sageattention-rdna3) - included in this image, compiled for gfx1151
- **RDNA4 (gfx120x)**: Use [jammm/SageAttention](https://github.com/jammm/SageAttention) `jam/gfx12-abi3` branch - does NOT work on Strix Halo
- **CDNA (Instinct)**: No community port yet

You can switch between backends at runtime:
```bash
# HIP native WMMA (faster for short sequences, default in this image)
docker exec -e SAGEATTN_BACKEND=native comfyui-gfx1151 ...

# Triton autotune (better compatibility, auto-dispatch fp16/int8)
docker exec -e SAGEATTN_BACKEND=triton comfyui-gfx1151 ...
```

To test SageAttention inside the container:
```bash
docker exec -it comfyui-gfx1151 python3 /opt/comfyui-gfx1151-utils/test-sageattention.py
```

## Get started now

The Docker image is published to both **Docker Hub** and **GitHub Container Registry**, so you can,
but don't have to build it yourself.

There are two options:

* Copy [docker-compose.yml](docker-compose.yml) and run `docker compose up -d`
* Copy [docker-run.sh](docker-run.sh) and run `./docker-run.sh`. After the first run, use `docker start comfyui-gfx1151`.

ComfyUI will be available at http://localhost:8188.
The starter templates should generate images without any issues.

Once you've verified that it works, feel free to use this repository as the foundation for your own setup or workflow.

### Pull from registries

```bash
# Docker Hub
docker pull arczewski/comfyui-gfx1151:latest

# GitHub Container Registry
docker pull ghcr.io/arczewski/comfyui-gfx1151:latest
```

#### Parameters

Both options have the same pre-configured parameters, which are:

* Allocate 8GB of shared memory (`shm_size`) for internal PyTorch / ComfyUI shenanigans, this should be plenty, 
  feel free to lower it. This should NOT be > than available RAM. This is NOT allocating VRAM.
* Mount `./ComfyUI` for the root of [ComfyUI](https://github.com/comfyanonymous/ComfyUI). If the directory is empty 
  when the container starts, it will copy a pre-cloned (baked in) version of ComfyUI. If it's not empty, it will be 
  used to run ComfyUI located in it. You can update this directory manually to use newer version of ComfyUI without 
  having to re-download the image
* Expose port `8188` for ComfyUI
* Add video + rendering devices and groups. While this just works on Arch, it might require some pre-requisite steps 
  on Ubuntu, I haven't checked.

There are a couple of scripts that can check that both PyTorch and flash-attention work, you can find them below.

#### Updating ComfyUI / dependencies

If you need to install custom nodes or refresh ComfyUI dependencies after a manual update,
you can do it from within the container (until #4 is resolved):

```bash
docker exec -it comfyui-gfx1151 /bin/bash

cd /opt/ComfyUI

pip install -r requirements.txt
```

## What's inside / how to replicate

This image is based on AMD's [rocm/pytorch](https://hub.docker.com/r/rocm/pytorch) image that has Ubuntu 24.04, 
ROCm 7.2, Python 3.12 and PyTorch 2.9.1, in which everything is configured to work together and it just works.
You can find out more about this image in 
[AMD's ROCm documentation](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/native_linux/install-pytorch.html#use-docker-image-with-pre-installed-pytorch).

There are only two missing pieces which this image adds: [flash-attention](https://github.com/ROCm/flash-attention/) 
and, well, ComfyUI.

There's nothing specific about the ComfyUI installation, you can actually bring your own, it should work.

**flash-attention**, however, "doesn't work" out of the box if you run AMD's image. I'm saying "doesn't work" because, 
as far as I understand, it doesn't have the frontend for it (the APIs), but it does have the backend: **Triton**. 
So flash-attention can be "installed" with a special env variable `FLASH_ATTENTION_TRITON_AMD_ENABLE`, which makes 
ComfyUI and other tools using flash-attention think that flash-attention is installed and works (even though it's 
triton under the hood, which is actually doing the job). You can see the lines that install it in 
[Dockerfile](Dockerfile), and if you try to do it yourself, you'll notice that it executes very fast 
(because flash-attention isn't actually built in full).

It's worth noting that flash-attention is cloned from a specific branch `main_perf` -- I'm not sure why exactly, 
I haven't checked, but I assume it's because it has (stable?) support for Triton which is not yet in the main branch, 
see [this issue](https://github.com/ROCm/flash-attention/issues/27). I basically copy-pasted this part from other
installations ([vLLM](https://community.frame.work/t/compiling-vllm-from-source-on-strix-halo/77241) and repos by 
[kyuz0](https://github.com/kyuz0)), so I hope they know what they're doing :D

In [scripts](scripts) there are two scripts that can check if PyTorch and flash-attention work as expected and utilize 
the iGPU. I used these when looking for a solution, they proved to be helpful, so I'm adding them to the image in case 
something breaks or doesn't work as expected, maybe they'll help debug the problem or something.

With that knowledge, you should be able to take [Dockerfile](Dockerfile) and build an image yourself.

If any of this makes more sense to you than it does to me and you know how to improve something or can add a helpful 
comment with additional context, please do!

## What I tried that didn't work

The majority of other solutions seem to rely on custom-built wheels, such as the image by 
[pccr10001/comfyui-gfx1151-fa](https://github.com/pccr10001/comfyui-gfx1151-fa).

I never managed to make these custom wheels work, presumably because of non-locked dependencies (pulling in newer 
version of rocm/etc with old wheels). However, they made me begin to understand what was happening and how to move 
forward, so huge thanks to everyone who left any comments on the topic.

Some other solutions also relied on the image `ghcr.io/rocm/therock_pytorch_dev_ubuntu_24_04_gfx1151`, which is no 
longer published, so I never got that working either. The image I'm referencing 
([rocm/pytorch](https://hub.docker.com/r/rocm/pytorch)) seems like a replacement for it though?

Initially, I [copied over](https://github.com/pccr10001/comfyui-gfx1151-fa/blob/e6e59be08ff439ab5f9799aa2161f70709fcd975/README.md?plain=1#L33)
some environment variables that were supposed to speed up ComfyUI / PyTorch and make it more resource efficient:
`PYTORCH_TUNABLEOP_ENABLED`, `MIOPEN_FIND_MODE` and `ROCBLAS_USE_HIPBLASLT` (not adding them as a codeblock to avoid
someone copy-pasting them). However, at least one of them not only made it worse when it comes to the speed, but I
believe it would crash my display server (X11) every now and then when running stable diffusion models. Apparently,
this is relatively common to see with AMD drivers in general, so I'm not entirely sure that those env variables were
100% responsible for the crashes (might've been something else), but removing all of them helped (at least for now),
so I've removed them from this repo's scripts too. If you also experience display server crashes, let me know.

## Tests

There are three scripts that you can use to test if everything works correctly

#### Test PyTorch

While the container is running, running

```bash
docker exec -it comfyui-gfx1151 /bin/bash /opt/comfyui-gfx1151-utils/test-pytorch.sh
```

should produce NO errors. The output should be something like:

```text
GPU: AMD Radeon Graphics | FlashAttn: True
Mean: -0.026233481243252754
```

#### Test flash-attention

While the container is working, running

```bash
docker exec -it comfyui-gfx1151 python3 /opt/comfyui-gfx1151-utils/test-pytorch-flashattention.py
```

#### Test SageAttention

```bash
docker exec -it comfyui-gfx1151 python3 /opt/comfyui-gfx1151-utils/test-sageattention.py
```

Expected output:

```text
=== SageAttention RDNA3 Check ===
PyTorch version: 2.9.1+rocm7.1.1.git351ff442
ROCm version: 7.1.52802-26aae437f6
CUDA available: True
Device: AMD Radeon Graphics

✓ sageattention imported successfully
  SAGEATTN_BACKEND = native

=== Forward Pass Test ===
✓ Forward pass OK — output shape: torch.Size([2, 8, 4096, 64]), dtype: torch.float16
  Mean: 0.001234, Std: 0.987654

=== Causal Mask Test ===
✓ Causal forward pass OK — output shape: torch.Size([1, 8, 128, 64])

=== All checks complete ===
```

should produce NO errors. The output should be something like:

```text
=== PyTorch Installation Check ===
PyTorch version: 2.9.1+rocm7.1.1.git351ff442
PyTorch ROCm version: 7.1.52802-26aae437f6
CUDA available: True
Device count: 1
Device name: AMD Radeon Graphics

=== Flash Attention Support Check ===
/usr/lib/python3.12/contextlib.py:105: FutureWarning: `torch.backends.cuda.sdp_kernel()` is deprecated. In the future, this context manager will be removed. Please see `torch.nn.attention.sdpa_kernel()` for the new context manager, with updated signature.
  self.gen = func(*args, **kwds)
Available SDP backends: <contextlib._GeneratorContextManager object at 0x7f9f4568c1d0>
Flash Attention backend enabled
Test tensors created on cuda
Flash Attention test successful! Output shape: torch.Size([2, 8, 128, 64])

=== AOTriton Check ===
AOTriton not available: No module named 'pyaotriton'

=== Environment Variables ===
ROCM_PATH: Not set
HIP_PATH: Not set
HIP_PLATFORM: Not set
HIP_ARCH: Not set
HSA_OVERRIDE_GFX_VERSION: Not set

=== Testing GFX Version Override ===
Set HSA_OVERRIDE_GFX_VERSION=11.0.0 to test gfx110x mapping
✓ Flash Attention worked with GFX override!

=== Flash Attention Backend Detection ===
✓ flash backend works
✓ mem_efficient backend works
✓ math backend works
```

## CI/CD

On every push to `develop` or `master`, GitHub Actions automatically:
1. Builds the Docker image
2. Pushes to both Docker Hub and GHCR
3. On `develop` builds, auto-merges into `master` (fast-forward) after a successful build
4. Sends a push notification with build status

See [.github/workflows/docker-publish.yml](.github/workflows/docker-publish.yml) for the workflow.

Required secrets:
| Secret | Description |
|---|---|
| `DOCKERHUB_USERNAME` | Docker Hub username |
| `DOCKERHUB_TOKEN` | Docker Hub access token (not password) |
| `PUSHOVER_TOKEN` | Pushover app token for build notifications |
| `PUSHOVER_USER` | Pushover user key for notifications |

## Acknowledgements

Big thanks to [pccr10001](https://github.com/pccr10001), [lhl](https://github.com/lhl), [kyuz0](https://github.com/kyuz0),
and [LuXuxue](https://github.com/LuXuxue) (SageAttention RDNA3) for setting me on the right path!

