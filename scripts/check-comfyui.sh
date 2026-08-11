#!/bin/bash

# -------------------------------------------------------------------
# 1. Initialize the persistent venv (first run / after host mount is wiped)
# -------------------------------------------------------------------
if [ ! -f /opt/venv/bin/python ]; then
    echo "No venv found at /opt/venv, extracting from built-in template..."
    if [ -f /opt/venv-template.tar.gz ]; then
        tar xzf /opt/venv-template.tar.gz -C /opt
        echo "Venv extracted successfully."
    else
        echo "ERROR: /opt/venv-template.tar.gz not found! Creating empty venv..."
        python3 -m venv /opt/venv --system-site-packages
        pip install -r /opt/ComfyUI-pre-cloned/requirements.txt 2>/dev/null || true
    fi
fi

# -------------------------------------------------------------------
# 2. If the user mounts /opt/ComfyUI, but their directory doesn't have
#    ComfyUI cloned, copy the pre-cloned one into the mounted directory.
# -------------------------------------------------------------------
if [ ! -e /opt/ComfyUI/requirements.txt ]; then
   echo "No ComfyUI detected, copying a built-in (pre-cloned) one..."
   cp -r /opt/ComfyUI-pre-cloned/{.,}* /opt/ComfyUI/
fi
