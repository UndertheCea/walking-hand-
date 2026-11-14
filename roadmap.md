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

With a planar floating palm and two fingers, the hand rests on one fingertip whenever the other finger swings. That is a point contact, so the palm pitches over unless something else supports it. We need to pick one before building the model:

- **Palm heel drags on the ground (recommended).** The wrist is a passive third support and the fingers pull the hand forward. It is statically stable, so the fixed-gait baseline can work.
- **Dynamic gait with no drag.** Much harder. The open-loop baseline will probably fall, which makes the comparison less informative.

Decision: _not yet made_

## Phases

### Phase 1: Model and environment (~1 week)

- [ ] Choose the hand model: Shadow Hand from MuJoCo Menagerie, or a simplified hand-written MJCF. Check whether the chosen model already couples the distal joints (we assume two actuated coordinates per active finger).
- [ ] Planar setup: palm has x-slide, z-slide and pitch-hinge joints only.
- [ ] Lock the thumb, ring and little fingers in a folded pose.
- [ ] Torque actuators with joint and torque limits. The impedance law is computed in Python so all three controllers share one low-level path.
- [ ] Fingertip contact-force sensors.
- [ ] Settable floor friction.
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
- [ ] Action: four joint targets (or offsets), passed through the same impedance layer and limits as the baseline.
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
