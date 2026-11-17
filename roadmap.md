# Walking Hand: Project Roadmap

Team: Nicole Shen, Tien-Hao Chen, Cheng-Yi Chen
Drafted: 2026-11-13
Final deadline: TBD (fill in; phase lengths below assume about four weeks)

## Goal

Make a five-finger robotic hand walk in MuJoCo using only its index and middle fingers, and compare three controllers on the same model and test conditions:

1. Fixed-gait joint-space impedance control (classical baseline)
2. Pure RL (PPO, no tactile observations)
3. Tactile-aware RL (PPO, same architecture and actions, plus fingertip contact forces)

The question we want to answer: how much does tactile feedback help learned walking when friction changes and the hand is pushed?

## Open decision: single-finger support

With two fingers, the hand rests on one fingertip whenever the other finger swings. That is a point contact, so the palm tips over unless something else supports it. The Thing hand already shows this: started in its `stand` keyframe with the fingers holding position, it falls onto its side within about 2 s. We need to pick one:

- **Palm heel or forearm stub drags on the ground.** The wrist is a passive third support and the fingers pull the hand forward. Optional via `stub_contact=True` / `STUB_CONTACT`.
- **Planar base + two fingertips (chosen for stand).** Slide x, slide z, hinge pitch — matches the proposal and makes double-support statically stable without a stub. Open-loop `stand` holds ≥20 s with the palm ~6–8 cm up.

Decision: **Planar base; two-finger stand by default** (`stub_contact=False`). Stub remains available as an option.

## Phases

### Phase 1: Model and environment (~1 week)

- [x] Hand model: the "Thing" hand (`models/thing_hand/`, MJCF built from the MakeHuman mesh). Index and middle each have 3 actuated joints (MCP abduction, MCP flexion, PIP flexion), and DIP is coupled to PIP at 0.7.
- [x] MuJoCo 3.17.0 in Docker, with the interactive viewer and offscreen rendering.
- [x] Base: planar (x-slide, z-slide, pitch hinge). Free 3D base dropped because two point contacts cannot hold open-loop in 3D.
- [x] Thumb, ring and little fingers fixed in a curled pose.
- [ ] Actuators: the model currently uses MuJoCo position actuators (`kp=8`, `kv=0.15`, force limit ±1). For the baseline's impedance law to run in Python, switch to torque (`motor`) actuators with the same limits so all three controllers share one low-level path.
- [x] Fingertip contact-force sensors (`index_touch`, `middle_touch`).
- [x] Settable floor friction (`floor` geom in `scene.xml`).
- [x] Gymnasium env (`walking_hand/`, `WalkingHand-v1` / `WalkingHandTactile-v1`, with v0 aliases): reset, step, fall check (palm height and tilt), friction randomization, external pushes.
- [x] A statically stable two-finger `stand` keyframe (CoM between tips, planar base, straighter PIP). Open-loop hold lasts the full 20 s episode with palm ~10 cm up.
- [ ] Agree on the env interface (observation, action, info fields) in the first few days so the other phases can start in parallel.

### Phase 2: Classical baseline (~1 week, overlaps Phase 1)

- [ ] Phase clock driving stance and swing fingertip trajectories in the palm frame, index and middle half a cycle apart.
- [ ] Inverse kinematics for the coupled finger, clamped to joint limits.
- [ ] Impedance control: `tau = K (q_des - q) - D qdot`, clipped to torque limits.
- [ ] Hand-tune step length, step height, frequency, K and D until it walks on nominal friction.
- [ ] No use of tactile measurements in this controller.

### Phase 3: RL training (~1.5 weeks)

- [ ] PPO from Stable-Baselines3 with parallel CPU envs.
- [ ] Action: six joint targets (or offsets), passed through the same impedance layer and limits as the baseline.
- [ ] Pure RL observations: joint positions and velocities, palm height, pitch and their velocities, previous action.
- [ ] Tactile RL observations: the same, plus fingertip contact forces. Nothing else differs.
- [ ] Reward: forward velocity and alive bonus, minus penalties on torque, action rate and palm pitch.
- [ ] Training randomization: floor friction and small pushes, identical for both policies.
- [ ] Same training budget (environment steps) and at least three seeds per policy.

### Phase 4: Evaluation (~0.5 week)

- [ ] Conditions: nominal flat ground; friction sweep including values outside the training range; pushes of varying magnitude at random times.
- [ ] 50 to 100 episodes per controller per condition, fixed evaluation seeds.
- [ ] Metrics:
  - forward distance
  - mean walking speed
  - fall rate
  - fingertip slip (tangential travel while in contact)
  - disturbance recovery (success rate, time to return to nominal speed after a push)
- [ ] One script that writes a results CSV, plots and a few rendered videos.

### Phase 5: Report and presentation (~0.5 week)

- [ ] Comparison plots across the three controllers.
- [ ] Failure cases for each controller.
- [ ] Discussion of what tactile feedback did and did not improve.

## Risks

- **Small tactile gap.** The pure-RL policy can partly infer contact from joint tracking error, so the difference may only appear at low friction and under pushes. Test conditions should stress those cases.
- **Reward exploits.** Policies may hop, drag or vibrate forward instead of walking. Leave time for reward iteration.
- **Contact tuning.** Fingertip geometry and solver parameters strongly affect slip. Fix them early and keep them identical across controllers.
- **Unfair comparison.** Any change to limits, randomization or training budget must apply to both RL policies; the baseline must use the same model and actuator limits.

## Suggested ownership

| Area | Owner |
| --- | --- |
| Hand model and environment | TBD |
| Baseline gait and impedance controller | TBD |
| RL training and evaluation harness | TBD |

## Planned stack

Python, MuJoCo, Gymnasium, Stable-Baselines3 (PPO), NumPy, Matplotlib.

## Workflow

See `CLAUDE.md`. In short: never push to `main`; env changes go through their own `[ENV]` PR and are logged below; training branches rebase onto `main` after every env change. CI (GitHub Actions) runs the model and env checks on every PR, enforces the `[ENV]` rules, and deletes branches after merge.

## Env changelog

Newest first. Every env change adds an entry here (see `CLAUDE.md`, rule 3).

| Date | Env version | Change | Action for training branches |
| --- | --- | --- | --- |
| 2026-11-17 | `WalkingHand-v1`, `WalkingHandTactile-v1` (v0 aliases kept) | Planar base (x, z, pitch). Tall two-finger `stand` (straighter PIP, CoM between tips); open-loop hold ≥20 s, palm ~10 cm (`stub_contact=False`). Search: `scripts/search_two_finger_stand.py`. | Rebase onto new `main`. Retrain / retune; freejoint-based results are not comparable. Prefer env ids `*-v1`. |
| 2026-11-16 | `WalkingHand-v0`, `WalkingHandTactile-v0` | First version: Thing hand, MuJoCo 3.17.0, 6 position actuators, 28-D obs (+2 tactile), forward-velocity reward, fall termination, friction and push options (commit `3fc4138`). | None (first version). |
