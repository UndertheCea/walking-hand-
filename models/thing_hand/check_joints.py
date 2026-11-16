import mujoco, numpy as np
for sc in ["scene.xml", "scene_inspect.xml"]:
    m = mujoco.MjModel.from_xml_path(sc)
    print(f"== {sc}: nq={m.nq} nv={m.nv} nu={m.nu} gravity={m.opt.gravity} mass={1000*m.body_mass.sum():.0f} g")
print("joints:", [m.joint(i).name for i in range(m.njnt)])
print("actuators:", [(m.actuator(i).name, m.actuator_ctrlrange[i].round(2).tolist()) for i in range(m.nu)])
d = mujoco.MjData(m)
def cons():
    return sorted({m.geom(d.contact[i].geom1).name+'|'+m.geom(d.contact[i].geom2).name for i in range(d.ncon)})
qa = lambda n: m.jnt_qposadr[m.joint(n).id]
tip = lambda f: d.site_xpos[m.site(f+'_tip').id].copy()
def setq(**kw):
    mujoco.mj_resetData(m, d)
    for k, v in kw.items(): d.qpos[qa(k)] = v
    for f in ["index","middle"]: d.qpos[qa(f+"_dip")] = 0.7*d.qpos[qa(f+"_pip")]
    mujoco.mj_forward(m, d)
setq(); print("all zero: contacts", cons())
mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d); print("stand_pose: contacts", cons())
setq(); t0 = tip("index")
setq(index_abd=0.3); print("index_abd +0.3: tip moves (world) mm", (1000*(tip("index")-t0)).round(1), "(+y = thumb side)")
setq(index_mcp=1.0); t1 = tip("index"); setq(index_abd=0.3, index_mcp=1.0); print("flex with abd: lateral offset grows mm", (1000*(tip("index")-t1)).round(1))
setq(index_abd=-0.35, middle_abd=0.35); print("fingers wag toward each other: contacts", cons())
# slider test: drive ctrl in zero-g fixed base and check each joint reaches the target
for i in range(m.nu):
    mujoco.mj_resetData(m, d); d.ctrl[:] = 0
    j = m.actuator_trnid[i, 0]; lo, hi = m.actuator_ctrlrange[i]
    tgt = lo + 0.7*(hi-lo); d.ctrl[i] = tgt
    for _ in range(500): mujoco.mj_step(m, d)
    print(f"  slider {m.actuator(i).name:15s} -> target {tgt:+.2f} reached {d.qpos[m.jnt_qposadr[j]]:+.3f} rad   contacts {cons()}")
