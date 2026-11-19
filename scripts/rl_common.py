"""Helpers shared by the RL training and evaluation scripts.

Not part of the env: nothing here changes what a policy sees or is rewarded
for. It builds envs from a saved config, records provenance (env id, git
commit, library versions) and loads trained PPO policies for evaluation.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    # Lets `python scripts/xxx.py` find the walking_hand package without
    # PYTHONPATH (Docker and CI set it; a plain conda shell does not).
    # Spawned worker processes on Windows re-run this module, so they get it too.
    sys.path.insert(0, str(REPO_ROOT))


def parse_kv(pairs: list[str] | None) -> dict:
    """Turn ["alive_bonus=0.2", "tactile_scale=2"] into a dict, parsing values as JSON when possible."""
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise ValueError(f"expected key=value, got {p!r}")
        k, v = p.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def git_info() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:
            return ""

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
    }


def versions() -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ["mujoco", "gymnasium", "stable_baselines3", "torch", "numpy"]:
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    return out


def provenance() -> dict:
    return {"created": datetime.now().isoformat(timespec="seconds"), "git": git_info(), "versions": versions()}


def json_ready(obj):
    """Make tuples, numpy scalars and arrays JSON-serialisable."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def save_json(path: Path, data: dict):
    path.write_text(json.dumps(json_ready(data), indent=2), encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def to_tuple_ranges(kwargs: dict) -> dict:
    """JSON turns tuples into lists; the env expects tuples for *_range arguments."""
    return {k: (tuple(v) if k.endswith("_range") and isinstance(v, list) else v) for k, v in kwargs.items()}


def make_env(env_id: str, env_kwargs: dict | None = None, render_mode: str | None = None):
    import gymnasium as gym

    import walking_hand  # noqa: F401  (registers the envs)

    kw = to_tuple_ranges(dict(env_kwargs or {}))
    if render_mode is not None:
        kw["render_mode"] = render_mode
    return gym.make(env_id, **kw)


def find_model(run_dir: Path, which: str = "best") -> tuple[Path, Path | None]:
    """Return (model.zip, vecnormalize.pkl or None) for a training run.

    which: "best" (EvalCallback's best model, falling back to final), "final",
    or a path to a specific checkpoint .zip.
    """
    run_dir = Path(run_dir)
    if which not in ("best", "final"):
        model = Path(which)
        stem = model.stem  # rl_model_100000_steps -> rl_model_vecnormalize_100000_steps.pkl
        cand = model.with_name(stem.replace("rl_model_", "rl_model_vecnormalize_") + ".pkl")
        return model, (cand if cand.exists() else _first_existing(run_dir / "vecnormalize.pkl"))
    if which == "best" and (run_dir / "best" / "best_model.zip").exists():
        vn = _first_existing(run_dir / "best" / "vecnormalize.pkl", run_dir / "vecnormalize.pkl")
        return run_dir / "best" / "best_model.zip", vn
    return run_dir / "final_model.zip", _first_existing(run_dir / "vecnormalize.pkl")


def _first_existing(*paths):
    for p in paths:
        if p is not None and Path(p).exists():
            return Path(p)
    return None


class PPOPolicy:
    """Wrap a saved SB3 PPO model (plus its VecNormalize obs statistics) as a controller."""

    def __init__(self, model_path: Path, vecnorm_path: Path | None, deterministic: bool = True):
        import pickle

        from stable_baselines3 import PPO

        self.model = PPO.load(str(model_path), device="cpu")
        self.deterministic = deterministic
        self.obs_rms, self.clip_obs, self.epsilon = None, 10.0, 1e-8
        if vecnorm_path is not None:
            # Read only the statistics so no dummy VecEnv is needed.
            with open(vecnorm_path, "rb") as f:
                vn = pickle.load(f)
            if getattr(vn, "norm_obs", False):
                self.obs_rms = vn.obs_rms
                self.clip_obs = vn.clip_obs
                self.epsilon = vn.epsilon

    def reset(self, env):
        pass

    def act(self, obs, env=None):
        import numpy as np

        if self.obs_rms is not None:
            obs = np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon),
                          -self.clip_obs, self.clip_obs)
        action, _ = self.model.predict(obs.astype(np.float32), deterministic=self.deterministic)
        return action


class HoldPolicy:
    """Hold the reset pose: the "do nothing" reference controller."""

    def reset(self, env):
        self.action = env.unwrapped.prev_action.copy()

    def act(self, obs, env=None):
        return self.action


class RandomPolicy:
    def reset(self, env):
        self.space = env.action_space

    def act(self, obs, env=None):
        return self.space.sample()
