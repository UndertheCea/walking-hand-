"""Classical baseline controllers.

Stand (balance only): ``BalanceHoldController`` → ``scripts/run_balance.py``
Walk (short-step gait): ``WalkController`` → ``scripts/run_walk.py``
"""

from walking_hand.baseline.balance import BalanceHoldController, STAND_CTRL, ctrl_to_action
from walking_hand.baseline.gait import WalkController
from walking_hand.baseline.ik import FingerIK

# Alias: stand == balance hold (no gait clock).
StandController = BalanceHoldController

__all__ = [
    "BalanceHoldController",
    "StandController",
    "WalkController",
    "FingerIK",
    "STAND_CTRL",
    "ctrl_to_action",
]
