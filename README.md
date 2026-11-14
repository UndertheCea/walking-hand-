# Walking Hand

Making a five-finger robotic hand (Inspire RH56) walk on its index and middle fingers. We compare a fixed-gait impedance controller against pure RL and tactile-aware RL (PPO). The plan and progress are in [roadmap.md](roadmap.md).

## Repository layout

```
docker/Dockerfile                      Gazebo Harmonic on Ubuntu 24.04
docker-compose.yml                     builds the image and opens the Gazebo window
gazebo/models/inspire_hand_right/      the hand as a Gazebo model (upstream URDF + meshes)
gazebo/worlds/hand_display.sdf         display scene: ground, light, hand
roadmap.md                             project plan
```

## Show the hand in Gazebo

Requirements: Docker Desktop. On Windows 11 the window is shown through WSLg, which is built in, so no X server needs to be installed.

```bash
docker compose build      # first time only, ~10 min
docker compose up         # opens the Gazebo window
```

The world starts paused, so the hand stays upright in its zero pose. The hand is not fixed to anything, so pressing play lets it fall over onto the ground.

Close the window or press `Ctrl+C` to stop. Run `docker compose down` to remove the container.

Linux host: run `xhost +local:docker` once, then `X11_DIR=/tmp/.X11-unix docker compose up`.

To get a shell in the container (for `gz` CLI commands or Python bindings):

```bash
docker compose run --rm gazebo bash
```

## Hand model

`gazebo/models/inspire_hand_right/inspire_hand_right.urdf` is copied unchanged from [dex-urdf](https://github.com/dexsuite/dex-urdf) (`robots/hands/inspire_hand/inspire_hand_right.urdf`, commit `f5e7132`, MIT licence, see `LICENSE.txt` in that folder), together with the right-hand meshes. Gazebo resolves its relative `meshes/...` paths against the model folder.

The world uses the bullet-featherstone physics engine rather than Gazebo's default DART. DART ignores the hand's collision meshes because they have no vertex normals, and it does not enforce the URDF mimic joints.

Joints: 6 actuated (`thumb_proximal_yaw`, `thumb_proximal_pitch`, `index_proximal`, `middle_proximal`, `ring_proximal`, `pinky_proximal`). Each finger's `*_intermediate_joint` mimics its proximal joint. For walking we only use `index_proximal_joint` and `middle_proximal_joint`, with their intermediate joints coupled.
