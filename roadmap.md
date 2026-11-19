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
- [x] Gymnasium env (`walking_hand/`, `WalkingHand-v0` and `WalkingHandTactile-v0`): reset, step, fall check (palm height and tilt), friction randomization, external pushes.
- [ ] A statically stable start pose. Holding the current `stand` keyframe, the hand falls within about 0.3 s.
- [ ] Agree on the env interface (observation, action, info fields) in the first few days so the other phases can start in parallel.

### Phase 2: Classical baseline (~1 week, overlaps Phase 1)

- [ ] Phase clock driving stance and swing fingertip trajectories in the palm frame, index and middle half a cycle apart.
- [ ] Inverse kinematics for the coupled finger, clamped to joint limits.
- [ ] Impedance control: `tau = K (q_des - q) - D qdot`, clipped to torque limits.
- [ ] Hand-tune step length, step height, frequency, K and D until it walks on nominal friction.
- [ ] No use of tactile measurements in this controller.

### Phase 3: RL training (~1.5 weeks)

- [x] PPO from Stable-Baselines3 with parallel CPU envs. _Verified on v0 2026-11-19 (Windows, smoke runs)._ `scripts/train_ppo.py` (SubprocVecEnv, VecNormalize, best-model eval, checkpoints, TensorBoard). Each run saves env id, env kwargs, PPO settings, seed, git commit and library versions to `runs/<name>/config.json`. Extra deps in `requirements-train.txt`, kept out of `requirements.txt` so they do not count as an env change.
- [ ] Action: six joint targets (or offsets), passed through the same impedance layer and limits as the baseline.
- [x] Pure RL observations: joint positions and velocities, palm height, pitch and their velocities, previous action. _Already in the env (28-D: 6 joint angles, 6 joint velocities, gravity in palm frame 3, palm gyro 3, base velocity in palm frame 3, palm height 1, previous action 6); verified in the v0 smoke runs._
- [x] Tactile RL observations: the same, plus fingertip contact forces. Nothing else differs. _Already in the env (+2: `log1p` index and middle touch force); verified in the v0 smoke runs._
- [x] Reward: forward velocity and alive bonus, minus penalties on torque, action rate and palm pitch. _Already in the env; weights settable per run with `--env-kwarg`. Weights still need tuning (see Open issues: falling forward)._
- [ ] Training randomization: floor friction and small pushes, identical for both policies. _Wired up: `--friction-range` / `--push-range` in `train_ppo.py`; `run_matrix.py` passes the same arguments to both policies. Blocked on v1 and on the floor-friction check (Open issues)._
- [ ] Same training budget (environment steps) and at least three seeds per policy. _Tooling ready: `scripts/run_matrix.py --seeds 0 1 2 --total-steps N` trains and evaluates both policies with identical settings and writes `comparison.csv` / `comparison.png`. Tick once a full comparison has run on v1._

### Phase 4: Evaluation (~0.5 week)

- [x] Conditions: nominal flat ground; friction sweep including values outside the training range; pushes of varying magnitude at random times. _Defined in `scripts/eval_policy.py`: nominal; friction 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0; pushes 0.2 to 1.2 N every 1 to 3 s._ _Verified on v0 2026-11-19 (Windows, smoke runs)._
- [x] 50 to 100 episodes per controller per condition, fixed evaluation seeds. _`eval_policy.py` defaults to 50 episodes with seeds 1000 to 1049, the same for every controller and condition value._ _Verified on v0 2026-11-19 (Windows, smoke runs)._
- [ ] Metrics: _all five computed in `eval_policy.py`; slip = horizontal fingertip travel while touching on consecutive steps; recovery = no fall within 2 s after a push, and time until the 0.5 s average speed returns to 80% of the 1 s pre-push speed. Distance, speed, fall rate and slip verified on v0; push recovery not yet exercised because the v0 hand falls before the first push. Tick after a v1 run produces `pushes.csv` rows._
  - forward distance
  - mean walking speed
  - fall rate
  - fingertip slip (tangential travel while in contact)
  - disturbance recovery (success rate, time to return to nominal speed after a push)
- [x] One script that writes a results CSV, plots and a few rendered videos. _`scripts/eval_policy.py`: `episodes.csv`, `summary.csv` (mean, std, 95% CI), `pushes.csv`, `eval_summary.png`, `videos/*.mp4`. Evaluates PPO runs (`--run`) and any Python controller (`--controller module:factory`, plus `hold` and `random` references), so the baseline can use the same harness._ _Verified on v0 2026-11-19 (Windows, smoke runs)._

### Phase 5: Report and presentation (~0.5 week)

- [ ] Comparison plots across the three controllers.
- [ ] Failure cases for each controller.
- [ ] Discussion of what tactile feedback did and did not improve.

## Risks

- **Small tactile gap.** The pure-RL policy can partly infer contact from joint tracking error, so the difference may only appear at low friction and under pushes. Test conditions should stress those cases.
- **Reward exploits.** Policies may hop, drag or vibrate forward instead of walking. Leave time for reward iteration.
- **Contact tuning.** Fingertip geometry and solver parameters strongly affect slip. Fix them early and keep them identical across controllers.
- **Unfair comparison.** Any change to limits, randomization or training budget must apply to both RL policies; the baseline must use the same model and actuator limits.

## Open issues to verify

Found while reading the code; none of these has been confirmed by a run yet. Whoever confirms or rules one out should update its line.

- [ ] **CONFIRMED 2026-11-19 (v0): floor friction below 1.0 has no effect.** Measured by resetting with friction f and reading `data.contact[i].friction` for floor contacts: settings 0.1 / 0.3 / 0.5 / 1.0 all give effective 1.0; only 1.5 gives 1.5. `check_env.py` only prints the model's floor value, so it does not catch this. Original note: MuJoCo combines two geoms' friction by taking the maximum when their `priority` is equal. The finger collision geoms use friction 1.0 (`class="col"` in `build_hand.py`) and the floor has no `priority`, so `friction_range=(0.3, 1.5)` and `reset(options={"friction": 0.2})` may leave the effective contact friction at 1.0. Check: reset with friction 0.2 and print `data.contact[i].friction` for floor contacts. Likely fix (env change, own `[ENV]` PR): `priority="1"` on the floor geom. Must be settled before any friction training or evaluation.
- [ ] **Pushes lose their y component on a planar base.** `_apply_pushes` draws a random direction in the xy-plane. With a planar base (x, z, pitch), the y part does nothing, so the effective push is `|cos θ|` of the nominal magnitude (about 64% on average). Decide whether to push only along ±x.
- [ ] **Reward may favour falling forward. Seen on v0 (2026-11-19):** a 50k-step PPO smoke run shortened episodes from ~24 to ~6 steps while episode reward rose from -10 to +34 and forward velocity to 0.22 m/s, with every episode ending in a fall, i.e. it learned to lunge. v0 cannot stand, so this must be rechecked on v1 before tuning. Original note: At baseline speed (~1 cm/s), `20·vx` is about 0.2 per step while the alive bonus is 0.05, so lunging forward and ending the episode can pay more than walking slowly. Watch early PPO videos; if it happens, raise `alive_bonus` or cap `vx`. Reward weights are constructor arguments, so tuning them is not an env change, but both RL policies must use the same values.

## Suggested ownership

| Area | Owner |
| --- | --- |
| Hand model and environment | TBD |
| Baseline gait and impedance controller | TBD |
| RL training and evaluation harness | Tien-Hao Chen |

## Planned stack

Python, MuJoCo, Gymnasium, Stable-Baselines3 (PPO), NumPy, Matplotlib.

## Workflow

See `CLAUDE.md`. In short: never push to `main`; env changes go through their own `[ENV]` PR and are logged below; training branches rebase onto `main` after every env change. CI (GitHub Actions) runs the model and env checks on every PR, enforces the `[ENV]` rules, and deletes branches after merge.

## Env changelog

Newest first. Every env change adds an entry here (see `CLAUDE.md`, rule 3).

| Date | Env version | Change | Action for training branches |
| --- | --- | --- | --- |
| 2026-11-16 | `WalkingHand-v0`, `WalkingHandTactile-v0` | First version: Thing hand, MuJoCo 3.17.0, 6 position actuators, 28-D obs (+2 tactile), forward-velocity reward, fall termination, friction and push options (commit `3fc4138`). | None (first version). |
