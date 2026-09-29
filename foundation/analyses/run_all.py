"""Regenerate every analysis result JSON (CPU only; about two minutes).

Outputs land in results/revision/runs/analyses/.
"""
import runpy
import os

HERE = os.path.dirname(os.path.abspath(__file__))
for s in ("cohort_stats.py", "resp_analyses.py", "event_true.py", "stats_analyses.py",
          "param_counts.py", "sleepedf_inventory.py", "edf_channels.py", "repr_block.py"):
    print("==", s)
    runpy.run_path(os.path.join(HERE, s), run_name="__main__")
