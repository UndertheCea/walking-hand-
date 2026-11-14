# Gazebo Harmonic (LTS) on Ubuntu 24.04.
# Build and run through docker-compose.yml at the repo root.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg lsb-release \
    && curl -fsSL https://packages.osrfoundation.org/gazebo.gpg \
        -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
        > /etc/apt/sources.list.d/gazebo-stable.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        gz-harmonic \
        python3-gz-sim8 python3-gz-transport13 python3-gz-msgs10 \
        libgl1-mesa-dri mesa-utils x11-apps \
        libqt5svg5 qml-module-qtquick-dialogs \
    && rm -rf /var/lib/apt/lists/*

# Models in gazebo/models are found by `model://<name>` URIs.
ENV GZ_SIM_RESOURCE_PATH=/workspace/gazebo/models
ENV XDG_RUNTIME_DIR=/tmp/runtime-root
RUN mkdir -p /tmp/runtime-root && chmod 700 /tmp/runtime-root

WORKDIR /workspace
CMD ["bash"]
