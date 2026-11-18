"""Deprecated alias: use run_balance.py (stand) or run_walk.py (walk)."""
from __future__ import annotations

import run_walk

if __name__ == "__main__":
    print("note: scripts/run_baseline.py now aliases walk; prefer scripts/run_walk.py")
    print("      stand / balance only → scripts/run_balance.py")
    run_walk.main()
