# Intentionally empty.
#
# We DO NOT re-export from `precompute_v1_stats` here, because doing so
# would trigger Python's "module imported twice" RuntimeWarning when the
# user runs `python -m mri_edain_v1.precompute.precompute_v1_stats`
# (first via this __init__'s import, then via __main__).
#
# Import the function explicitly when needed:
#     from mri_edain_v1.precompute.precompute_v1_stats import (
#         precompute_v1_stats_from_npz,
#     )
