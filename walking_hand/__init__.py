"""Walking Hand: Gymnasium environments for the Thing hand.

Importing this package registers:
    WalkingHand-v1          pure RL observations
    WalkingHandTactile-v1   same, plus fingertip touch forces
    WalkingHand-v0 / WalkingHandTactile-v0  aliases of v1 (kept so old scripts import)

Both default to 1000 steps (20 s at 50 Hz) per episode. Any constructor
argument of ThingHandWalkEnv can be passed to gymnasium.make as a keyword.
"""
from gymnasium.envs.registration import register

from walking_hand.envs import ThingHandWalkEnv

_COMMON = dict(
    entry_point="walking_hand.envs:ThingHandWalkEnv",
    max_episode_steps=1000,
)

register(id="WalkingHand-v1", **_COMMON, kwargs={"tactile": False})
register(id="WalkingHandTactile-v1", **_COMMON, kwargs={"tactile": True})
register(id="WalkingHand-v0", **_COMMON, kwargs={"tactile": False})
register(id="WalkingHandTactile-v0", **_COMMON, kwargs={"tactile": True})

__all__ = ["ThingHandWalkEnv"]
