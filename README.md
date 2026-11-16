# Walking Hand

Making a five-finger robotic hand walk on its index and middle fingers in MuJoCo. We compare a fixed-gait impedance controller against pure RL and tactile-aware RL (PPO). The plan and progress are in [roadmap.md](roadmap.md).

## Repository layout

```
docker/Dockerfile            Python 3.12 + MuJoCo 3.17.0, with GUI and offscreen rendering libraries
docker-compose.yml           builds the image and opens the MuJoCo viewer
requirements.txt             Python dependencies installed in the image
models/thing_hand/           the "Thing" hand model (MJCF) and its scenes
scripts/view.py              open a scene in the interactive viewer, starting from a keyframe
scripts/render.py            render a scene offscreen to a PNG
roadmap.md                   project plan
```

## Open the hand in the MuJoCo viewer

Requirements: Docker Desktop. On Windows 11 the window is shown through WSLg, which is built in, so no X server needs to be installed.

```bash
docker compose build      # first time only, a few minutes
docker compose up         # opens the viewer on models/thing_hand/scene.xml
```

The viewer starts in the `stand` keyframe with physics running. The hand cannot hold that pose by itself and tips over within about two seconds. Press `Backspace` to reset, or use the viewer's Simulation panel to pause and load keyframes.

To inspect the joints with the hand bolted in the air and gravity off, drag the sliders in the viewer's Control panel:

```bash
docker compose run --rm mujoco python scripts/view.py models/thing_hand/scene_inspect.xml
```

Linux host: run `xhost +local:docker` once, then prefix the commands with `X11_DIR=/tmp/.X11-unix`.

### Without a display

Render a frame to a PNG (here after 1 s of simulation):

```bash
docker compose run --rm -e MUJOCO_GL=osmesa mujoco \
    python scripts/render.py models/thing_hand/scene.xml out.png --seconds 1
```

Run the model's joint and actuator checks:

```bash
docker compose run --rm -w /workspace/models/thing_hand mujoco python check_joints.py
```

For a shell inside the container: `docker compose run --rm mujoco bash`.

## The Thing hand (`models/thing_hand/`)

A realistic right hand built from the MakeHuman base mesh (CC0, see `LICENSE_makehuman_assets_CC0.md`). The skinned visual mesh is embedded in the XML, so there are no separate mesh files.

| File | Contents |
| --- | --- |
| `thing_hand.xml` | the hand with a 6-DOF free base |
| `thing_hand_fixed.xml` | the same hand bolted 30 cm above the floor |
| `scene.xml` | floor + free hand + `stand` keyframe |
| `scene_inspect.xml` | floor + fixed hand, gravity off, `stand_pose` keyframe |
| `build_hand.py` | regenerates the four XML files from MakeHuman data (`python build_hand.py <makehuman/data> <out dir>`) |
| `check_joints.py` | joint, contact and actuator sanity checks |

Frame: x is the finger and walking direction, y the thumb side, z the back of the hand.

Actuated: index and middle fingers, each with MCP abduction, MCP flexion and PIP flexion, so 6 position actuators (`kp=8`, `kv=0.15`, force limit ±1). Each DIP joint is coupled to its PIP joint at 0.7 through an equality constraint. Thumb, ring and pinky are rigid and curled.

Sensors: fingertip touch (`index_touch`, `middle_touch`), joint positions, velocities and actuator forces for the 6 actuated joints, and a palm IMU (orientation, gyro, accelerometer, position).

`build_hand.py` is the source of the XML files. Change parameters there (masses, gains, `STUB_CONTACT`, coupling ratio) and regenerate rather than editing the XML by hand.
