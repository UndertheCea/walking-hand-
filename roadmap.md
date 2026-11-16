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

- **Palm heel or forearm stub drags on the ground (recommended).** The wrist is a passive third support and the fingers pull the hand forward. It is statically stable, so the fixed-gait baseline can work. In the Thing hand this is `STUB_CONTACT = True` in `build_hand.py`.
- **Dynamic gait with no drag.** Much harder. The open-loop baseline will probably fall, which makes the comparison less informative.

Decision: _not yet made_

## Phases

### Phase 1: Model and environment (~1 week)

- [x] Hand model: the "Thing" hand (`models/thing_hand/`, MJCF built from the MakeHuman mesh). Index and middle each have 3 actuated joints (MCP abduction, MCP flexion, PIP flexion), and DIP is coupled to PIP at 0.7.
- [x] MuJoCo 3.17.0 in Docker, with the interactive viewer and offscreen rendering.
- [ ] Base: the Thing hand has a 6-DOF free base (3D). Decide whether to keep it or constrain it to planar (x-slide, z-slide, pitch hinge) as in the proposal.
- [x] Thumb, ring and little fingers fixed in a curled pose.
- [ ] Actuators: the model currently uses MuJoCo position actuators (`kp=8`, `kv=0.15`, force limit ±1). For the baseline's impedance law to run in Python, switch to torque (`motor`) actuators with the same limits so all three controllers share one low-level path.
- [x] Fingertip contact-force sensors (`index_touch`, `middle_touch`).
- [x] Settable floor friction (`floor` geom in `scene.xml`).
- [ ] Gymnasium env: reset, step, fall check (palm height and pitch thresholds), hook for applying external pushes.
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
