"""Build a MuJoCo model of a realistic right hand from the MakeHuman (CC0) base mesh.

Usage: python -I build_hand.py <makehuman/data dir> <out dir>
Produces <out>/thing_hand.xml (model with skinned mesh) and <out>/scene.xml.
"""
import json, sys, os
import numpy as np

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
S = 0.1  # MakeHuman units are decimeters

# ---------- load mesh (body group only) ----------
V, F, group = [], [], None
for line in open(os.path.join(src, "3dobjs/base.obj")):
    if line.startswith("v "):
        V.append([float(x) for x in line.split()[1:4]])
    elif line.startswith("g "):
        group = line.split()[1]
    elif line.startswith("f ") and group == "body":
        idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
        for k in range(1, len(idx) - 1):
            F.append([idx[0], idx[k], idx[k + 1]])
V = np.array(V) * S
F = np.array(F)

skel = json.load(open(os.path.join(src, "rigs/default.mhskel")))
W = json.load(open(os.path.join(src, "rigs/default_weights.mhw")))["weights"]
J = {k: V[v].mean(0) for k, v in skel["joints"].items()}
bones = skel["bones"]
head = lambda b: J[bones[b]["head"]]
tail = lambda b: J[bones[b]["tail"]]

# ---------- hand frame: x = fingers, y = index side, z = dorsal ----------
wrist = head("wrist.R")
x = head("finger3-1.R") - wrist; x /= np.linalg.norm(x)
lat = head("finger2-1.R") - head("finger5-1.R")
y = lat - lat.dot(x) * x; y /= np.linalg.norm(y)
z = np.cross(x, y)
R = np.stack([x, y, z])          # world(mesh) -> hand frame
to_hand = lambda p: R @ (p - wrist)

# ---------- bone -> MuJoCo body mapping ----------
FINGERS = {"thumb": 1, "index": 2, "middle": 3, "ring": 4, "pinky": 5}
SEG = ["prox", "mid", "dist"]
bone2body = {}
for fname, i in FINGERS.items():
    for s in range(3):
        bone2body[f"finger{i}-{s+1}.R"] = f"{fname}_{SEG[s]}"
PALM_BONES = ["wrist.R", "metacarpal1.R", "metacarpal2.R", "metacarpal3.R", "metacarpal4.R",
              "lowerarm02.R", "lowerarm01.R"]
for b in PALM_BONES:
    bone2body[b] = "palm"

# ---------- per-vertex body weights ----------
nV = len(V)
bodies = ["palm"] + [f"{f}_{s}" for f in FINGERS for s in SEG]
bidx = {b: i for i, b in enumerate(bodies)}
Wb = np.zeros((nV, len(bodies)))
Wother = np.zeros(nV)
for bone, lst in W.items():
    for vid, w in lst:
        if bone in bone2body:
            Wb[vid, bidx[bone2body[bone]]] += w
        else:
            Wother[vid] += w
hand_w = Wb.sum(1)
Vh = np.array([to_hand(p) for p in V])
FOREARM_KEEP = 0.05  # metres of forearm to keep behind the wrist
sel = (hand_w > 0.5) & (Vh[:, 0] > -FOREARM_KEEP)
# keep only faces fully inside the selection
fmask = sel[F].all(1)
Fh = F[fmask]
used = np.unique(Fh)
remap = -np.ones(nV, int); remap[used] = np.arange(len(used))
Fh = remap[Fh]
Vs = Vh[used]
Ws = Wb[used]
Ws[:, bidx["palm"]] += Wother[used]          # leftover weight -> palm
Ws /= Ws.sum(1, keepdims=True)
print(f"hand mesh: {len(Vs)} verts, {len(Fh)} tris")

# ---------- skeleton geometry in hand frame ----------
H = {b: to_hand(head(b)) for b in bones if b.endswith(".R")}
T = {b: to_hand(tail(b)) for b in bones if b.endswith(".R")}

def seg_radius(body):
    """Mean distance of strongly-weighted vertices to the bone segment."""
    bone = [k for k, v in bone2body.items() if v == body][0]
    a, b = H[bone], T[bone]
    m = Ws[:, bidx[body]] > 0.7
    p = Vs[m]
    t = np.clip((p - a) @ (b - a) / np.dot(b - a, b - a), 0, 1)
    d = np.linalg.norm(p - (a + t[:, None] * (b - a)), axis=1)
    return float(np.median(d)) if len(d) else 0.008

f3 = lambda v: " ".join(f"{c:.5f}" for c in v)

# Fixed (rigid, no joint) poses for the unactuated fingers, rad: (prox, mid, dist)
PASSIVE = {"thumb": (-0.4, 0.0, 0.2), "ring": (1.45, 1.6, 1.0), "pinky": (1.45, 1.6, 1.0)}
ACTIVE = ["index", "middle"]
ACT_J = ["index_abd", "index_mcp", "index_pip", "middle_abd", "middle_mcp", "middle_pip"]
ABD_RANGE = 0.35   # rad (~20 deg) each side for the abduction joints

# ---- Mass budget (kg), set to match the planned hardware. ----
# Default: 4x Dynamixel XL330 in the palm (~18 g each), PLA shell, small controller + IMU, tethered power.
MASS = {
    "palm": 0.170,        # shell ~70 g + 4 motors ~72 g + electronics/cables ~28 g
    "stub": 0.020,        # forearm shell
    "active": (0.010, 0.006, 0.004),   # index/middle: prox (pulley, bearings), mid (linkage), dist (pad)
    "passive": (0.004, 0.003, 0.002),  # thumb/ring/pinky (rigid): hollow shells
}
STUB_CONTACT = False   # False: only fingertips (and fingers) touch the ground; True: forearm stub acts as a tail
# Planar base (x, z, pitch) makes two-fingertip double-support statically stable.
PLANAR_BASE = True
COUPLE = 0.7

def finger_xml(fname, i, indent):
    lines = []
    parent = np.zeros(3)
    opened = 0
    for s in range(3):
        bone = f"finger{i}-{s+1}.R"
        body = f"{fname}_{SEG[s]}"
        a, b = H[bone], T[bone]
        d = (b - a) / np.linalg.norm(b - a)
        if fname == "thumb":
            axis = np.cross(np.array([0, 0, 1.0]), d)
            axis = np.cross(d, axis) if s == 0 else axis   # CMC swings the thumb under the palm
        else:
            axis = np.cross(np.array([0, 0, 1.0]), d)
        axis /= np.linalg.norm(axis)
        r = max(min(seg_radius(body) * 0.9, 0.011), 0.006)
        pad = "  " * (indent + opened)
        if fname in ACTIVE:
            lines.append(f'{pad}<body name="{body}" pos="{f3(a - parent)}">')
        else:
            th = PASSIVE[fname][s]
            qv = np.r_[np.cos(th / 2), np.sin(th / 2) * axis]
            lines.append(f'{pad}<body name="{body}" pos="{f3(a - parent)}" quat="{" ".join(f"{c:.5f}" for c in qv)}">')
        jn = f"{fname}_{['mcp','pip','dip'][s]}" if fname != "thumb" else f"thumb_{['cmc','mcp','ip'][s]}"
        if fname in ACTIVE:
            rng = "-0.3 1.6" if s == 0 else "0 1.6"
            if s == 0:
                # abduction/adduction (side-to-side wag) about the palm normal; listed first so the
                # flexion axis turns with it (universal-joint MCP). + = toward the thumb.
                lines.append(f'{pad}  <joint name="{fname}_abd" axis="0 0 1" range="{-ABD_RANGE} {ABD_RANGE}" class="finger"/>')
            lines.append(f'{pad}  <joint name="{jn}" axis="{f3(axis)}" range="{rng}" class="finger"/>')
        end = (b - a) if s < 2 else (b - a) * 0.88
        mass = MASS["active" if fname in ACTIVE else "passive"][s]
        lines.append(f'{pad}  <geom name="{body}_col" class="col" type="capsule" size="{r:.4f}" fromto="0 0 0 {f3(end)}" mass="{mass}"/>')
        if s == 2 and fname in ACTIVE:
            tip = (b - a) * 0.88
            lines.append(f'{pad}  <site name="{fname}_tip" pos="{f3(tip)}" size="{r+0.004:.4f}" rgba="1 0 0 0.25"/>')
        parent = a
        opened += 1
    for k in reversed(range(opened)):
        lines.append("  " * (indent + k) + "</body>")
    return lines

# palm collision box from metacarpal region
mc = np.array([H[f"finger{i}-1.R"] for i in range(2, 6)])
palm_len = mc[:, 0].mean()
palm_w = np.ptp(mc[:, 1]) / 2 + 0.01
pm = Ws[:, 0] > 0.8
palm_pts = Vs[pm & (Vs[:, 0] > 0) & (Vs[:, 0] < palm_len)]
palm_th = (np.percentile(palm_pts[:, 2], 90) - np.percentile(palm_pts[:, 2], 10)) / 2
palm_zc = (np.percentile(palm_pts[:, 2], 90) + np.percentile(palm_pts[:, 2], 10)) / 2
palm_yc = mc[:, 1].mean()

body_lines = []
for fname, i in FINGERS.items():
    body_lines += finger_xml(fname, i, 4)

# ---------- skin ----------
bone_xml = []
for b in bodies:
    col = Ws[:, bidx[b]]
    ids = np.nonzero(col > 1e-4)[0]
    if b == "palm":
        bp = np.zeros(3)
    else:
        fname, seg = b.split("_")
        bp = H[f"finger{FINGERS[fname]}-{SEG.index(seg)+1}.R"]
    bone_xml.append(
        f'      <bone body="{b}" bindpos="{f3(bp)}" bindquat="1 0 0 0" '
        f'vertid="{" ".join(map(str, ids))}" vertweight="{" ".join(f"{w:.4f}" for w in col[ids])}"/>')


xml = f"""<!--
  "Thing" hand: realistic five-finger right hand, two actuated fingers (index, middle).
  Visual mesh: MakeHuman base mesh (CC0 1.0, makehumancommunity/makehuman), right hand + {FOREARM_KEEP*100:.0f} cm of forearm,
  deformed with MuJoCo <skin> using the MakeHuman default rig weights.
  Frame: x = finger direction / walking direction, y = thumb side, z = back of hand (up).
  Actuated: index & middle, each with MCP abduction (side-to-side), MCP flexion and PIP flexion (6 motors).
  DIP = {COUPLE} * PIP (linkage coupling). Base: 6-DOF free joint (3D, can roll/pitch/yaw);
  thing_hand_fixed.xml is the same hand bolted in the air (no base joint) for inspection.
  Unactuated: thumb, ring, pinky are rigidly fixed to the palm (no joints) in a curled pose.
-->
<mujoco model="thing_hand">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" cone="elliptic" impratio="10"/>

  <default>
    <default class="hand">
      <joint damping="0.05" armature="0.0002" frictionloss="0.005"/>
      <geom density="1100"/>
      <!-- Torque limits: set to the real motor torque x transmission ratio. -->
      <position kp="8" kv="0.15" forcerange="-1 1"/>
      <default class="finger"/>
      <default class="col">
        <geom group="3" rgba="0.2 0.6 0.9 0.4" friction="1.0 0.01 0.0001" solimp="0.9 0.95 0.001" solref="0.004 1"/>
      </default>
    </default>
  </default>

  <asset>
    <texture name="skin_tex" type="2d" builtin="flat" rgb1="0.87 0.70 0.59" width="8" height="8"/>
    <material name="skin" texture="skin_tex" rgba="1 1 1 1" specular="0.15" shininess="0.3"/>
  </asset>

  <worldbody>
    <body name="base" @BASE@ childclass="hand">
@ROOTJOINT@
      <body name="palm">
        <site name="imu" pos="{palm_len/2:.4f} {palm_yc:.4f} {palm_zc:.4f}" size="0.005"/>
        <geom name="palm_col" class="col" type="box" size="{palm_len/2:.4f} {palm_w:.4f} {palm_th:.4f}" pos="{palm_len/2:.4f} {palm_yc:.4f} {palm_zc:.4f}" mass="{MASS['palm']}"/>
        <!-- forearm stub: low friction; collides only if STUB_CONTACT=True in build_hand.py -->
        <geom name="stub" class="col" type="capsule" size="0.022" fromto="-0.005 {palm_yc:.4f} {palm_zc:.4f} {-FOREARM_KEEP+0.015:.4f} {palm_yc:.4f} {palm_zc:.4f}" friction="0.05 0.005 0.0001" mass="{MASS['stub']}"{'' if STUB_CONTACT else ' contype="0" conaffinity="0"'}/>
{chr(10).join(body_lines)}
      </body>
    </body>
  </worldbody>

  <deformable>
    <skin name="hand_skin" material="skin" inflate="0.0005"
      vertex="{" ".join(f"{c:.5f}" for c in Vs.ravel())}"
      face="{" ".join(map(str, Fh.ravel()))}">
{chr(10).join(bone_xml)}
    </skin>
  </deformable>

  <contact>
    <exclude body1="palm" body2="thumb_prox"/>
    <exclude body1="palm" body2="thumb_mid"/>
    <exclude body1="palm" body2="thumb_dist"/>
    <exclude body1="palm" body2="index_prox"/>
    <exclude body1="palm" body2="middle_prox"/>
    <exclude body1="palm" body2="ring_prox"/>
    <exclude body1="palm" body2="pinky_prox"/>
    <exclude body1="palm" body2="ring_mid"/>
    <exclude body1="palm" body2="pinky_mid"/>
    <exclude body1="palm" body2="ring_dist"/>
    <exclude body1="palm" body2="pinky_dist"/>
    <exclude body1="thumb_mid" body2="index_prox"/>
    <exclude body1="thumb_mid" body2="index_mid"/>
    <exclude body1="thumb_mid" body2="index_dist"/>
    <exclude body1="thumb_dist" body2="index_prox"/>
    <exclude body1="thumb_dist" body2="index_mid"/>
    <exclude body1="thumb_dist" body2="index_dist"/>
    <exclude body1="middle_prox" body2="ring_prox"/>
    <exclude body1="ring_prox" body2="pinky_prox"/>
    <exclude body1="ring_mid" body2="pinky_mid"/>
    <exclude body1="ring_dist" body2="pinky_dist"/>
    <exclude body1="middle_prox" body2="ring_mid"/>
    <exclude body1="middle_prox" body2="ring_dist"/>
    <exclude body1="middle_mid" body2="ring_mid"/>
    <exclude body1="middle_mid" body2="ring_dist"/>
  </contact>

  <equality>
    <joint name="index_couple" joint1="index_dip" joint2="index_pip" polycoef="0 {COUPLE} 0 0 0" solref="0.005 1"/>
    <joint name="middle_couple" joint1="middle_dip" joint2="middle_pip" polycoef="0 {COUPLE} 0 0 0" solref="0.005 1"/>
  </equality>

  <actuator>
    <position name="index_abd_act" joint="index_abd" class="hand" ctrlrange="{-ABD_RANGE} {ABD_RANGE}"/>
    <position name="index_mcp_act" joint="index_mcp" class="hand" ctrlrange="-0.3 1.6"/>
    <position name="index_pip_act" joint="index_pip" class="hand" ctrlrange="0 1.6"/>
    <position name="middle_abd_act" joint="middle_abd" class="hand" ctrlrange="{-ABD_RANGE} {ABD_RANGE}"/>
    <position name="middle_mcp_act" joint="middle_mcp" class="hand" ctrlrange="-0.3 1.6"/>
    <position name="middle_pip_act" joint="middle_pip" class="hand" ctrlrange="0 1.6"/>
  </actuator>

  <sensor>
    <touch name="index_touch" site="index_tip"/>
    <touch name="middle_touch" site="middle_tip"/>
{chr(10).join(f'    <jointpos name="{j}_q" joint="{j}"/>' for j in ACT_J)}
{chr(10).join(f'    <jointvel name="{j}_dq" joint="{j}"/>' for j in ACT_J)}
{chr(10).join(f'    <actuatorfrc name="{a}_tau" actuator="{a}_act"/>' for a in ACT_J)}
    <framequat name="palm_quat" objtype="site" objname="imu"/>
    <gyro name="palm_gyro" site="imu"/>
    <accelerometer name="palm_acc" site="imu"/>
    <framepos name="palm_pos" objtype="site" objname="imu"/>
  </sensor>
</mujoco>
"""
QPITCH = "0.92388 0 0.38268 0"   # palm pitched 45 deg nose-down (fixed inspection only)
PLANAR_ROOT = """      <joint name="root_x" type="slide" axis="1 0 0" damping="0"/>
      <joint name="root_z" type="slide" axis="0 0 1" damping="0"/>
      <joint name="root_pitch" type="hinge" axis="0 1 0" range="-1.2 1.2" damping="0.02"/>"""
if PLANAR_BASE:
    open(os.path.join(out, "thing_hand.xml"), "w").write(
        xml.replace("@BASE@", "").replace("@ROOTJOINT@", PLANAR_ROOT))
else:
    open(os.path.join(out, "thing_hand.xml"), "w").write(
        xml.replace("@BASE@", f'pos="0 0 0.122" quat="{QPITCH}"').replace("@ROOTJOINT@", '      <freejoint name="root"/>'))
open(os.path.join(out, "thing_hand_fixed.xml"), "w").write(
    xml.replace("@BASE@", f'pos="0 0 0.30" quat="{QPITCH}"').replace("@ROOTJOINT@", "      <!-- base fixed to the world (inspection) -->")
       .replace('<mujoco model="thing_hand">', '<mujoco model="thing_hand_fixed">'))

# Tall two-finger stand (search + ≥20 s hold with STUB_CONTACT=False, planar base):
# straighter PIP (extended legs), CoM between tips, palm ~10 cm off the floor.
# (abd, mcp, pip, dip)
stand = {"index": (-0.0727, 0.7260, 0.0632, 0.0632 * COUPLE), "middle": (-0.1129, 0.9953, 0.1237, 0.1237 * COUPLE)}
qf = []
for fname in FINGERS:
    if fname in stand: qf += list(stand[fname])
# Planar qpos: root_x, root_z, root_pitch, then finger joints in tree order.
q = [-0.0781, 0.1377, 0.8402] + qf
ctrl = [*stand["index"][:3], *stand["middle"][:3]]
scene = f"""<mujoco model="thing_hand scene">
  <include file="thing_hand.xml"/>
  <statistic center="0.05 0 0.05" extent="0.3"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35" specular="0 0 0"/>
    <global azimuth="120" elevation="-20"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0.3 -0.3 1.2" dir="-0.2 0.2 -1" directional="true"/>
    <!-- Floor friction (first number) is the knob for the friction experiments. -->
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" friction="1.0 0.005 0.0001"/>
  </worldbody>
  <keyframe>
    <key name="stand" qpos="{" ".join(f"{v:.4f}" for v in q)}" ctrl="{" ".join(f"{v:.4f}" for v in ctrl)}"/>
  </keyframe>
</mujoco>
"""
open(os.path.join(out, "scene.xml"), "w").write(scene)
# Inspection scene: hand bolted in the air, gravity off. Drag the Control sliders in the viewer.
inspect = (scene.replace('<include file="thing_hand.xml"/>', '<include file="thing_hand_fixed.xml"/>\n  <option gravity="0 0 0"/>')
                .replace('<mujoco model="thing_hand scene">', '<mujoco model="thing_hand inspect (fixed base, no gravity)">')
                .replace(" ".join(f"{v:.4f}" for v in q), " ".join(f"{v:.4f}" for v in qf))
                .replace('<key name="stand"', '<key name="stand_pose"')
                .replace('<statistic center="0.05 0 0.05" extent="0.3"/>', '<statistic center="0.06 0 0.27" extent="0.25"/>'))
open(os.path.join(out, "scene_inspect.xml"), "w").write(inspect)
print("palm len %.3f half-width %.3f half-thick %.3f" % (palm_len, palm_w, palm_th))
for f in ["index", "middle"]:
    i = FINGERS[f]
    print(f, "phalanx lengths (mm):", [round(1000 * np.linalg.norm(T[f"finger{i}-{s}.R"] - H[f"finger{i}-{s}.R"]), 1) for s in (1, 2, 3)])
