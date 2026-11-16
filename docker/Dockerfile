# MuJoCo + Python. Build and run through docker-compose.yml at the repo root.
FROM python:3.12-slim-bookworm

# OpenGL and X11 libraries for the interactive viewer (GLFW), plus OSMesa and
# EGL for offscreen rendering (MUJOCO_GL=osmesa or egl).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglx-mesa0 libgl1-mesa-dri libegl1 libosmesa6 \
        libx11-6 libxext6 libxrandr2 libxinerama1 libxcursor1 libxi6 \
        libxxf86vm1 libxkbcommon0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --root-user-action=ignore -r /tmp/requirements.txt

WORKDIR /workspace
CMD ["bash"]
