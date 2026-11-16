"""Walking Hand: Gymnasium environments for the Thing hand.

Importing this package registers:
    WalkingHand-v0          pure RL observations
    WalkingHandTactile-v0   same, plus fingertip touch forces
Both default to 1000 steps (20 s at 50 Hz) per episode. Any constructor
argument of ThingHandWalkEnv can be passed to gymnasium.make as a keyword.
"""
from gymnasium.envs.registration import register

from walking_hand.envs import ThingHandWalkEnv

register(
    id="WalkingHand-v0",
    entry_point="walking_hand.envs:ThingHandWalkEnv",
    max_episode_steps=1000,
    kwargs={"tactile": False},
)
register(
    id="WalkingHandTactile-v0",
    entry_point="walking_hand.envs:ThingHandWalkEnv",
    max_episode_steps=1000,
    kwargs={"tactile": True},
)

__all__ = ["ThingHandWalkEnv"]
