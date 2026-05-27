#!/bin/bash
#SBATCH --job-name=diagnose_edain_output
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu[005-006]
#SBATCH -o logs/slurm_%x_%j.out
#SBATCH -e logs/slurm_%x_%j.err

# ============================================================
# Load the trained Nyul-identity checkpoint and dump what the EDAIN layer
# actually OUTPUTS for: 2 working val cases (Lipo-001, Lipo-021) vs the
# 3 failure cases (Lipo-077, Lipo-089, Lipo-097).
#
# Compares against the baseline z-score (what the UNet was originally
# expected to see). Reveals whether EDAIN is producing distributions
# wildly different from training-time expectations.
# ============================================================

set -euo pipefail
module purge
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load Python/3.11.5-GCCcore-13.2.0
source ~/nnunet_env/bin/activate

export nnUNet_raw=${nnUNet_raw:-$HOME/nnUNet_data/raw}
export nnUNet_preprocessed=${nnUNet_preprocessed:-$HOME/nnUNet_data/preprocessed}
export nnUNet_results=${nnUNet_results:-$HOME/nnUNet_data/results}
export nnUNet_n_proc_DA=0
export nnUNet_compile=F

REPO_ROOT="$SLURM_SUBMIT_DIR/.."
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
python -m tools.register_trainers >/dev/null

python - <<'PY'
import os, json
from pathlib import Path
import numpy as np
import torch

import sys
sys.path.insert(0, os.environ['PYTHONPATH'].split(':')[0])

# ---------- Load the trained Nyul-identity wrapper ----------
from nnunetv2.training.nnUNetTrainer.variants.edain_register.nnUNetTrainerNyulIdentity \
    import nnUNetTrainerNyulIdentity   # registered by register_trainers
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from batchgenerators.utilities.file_and_folder_operations import join, load_json

dataset_name = maybe_convert_to_dataset_name(500)
plans = load_json(join(nnUNet_preprocessed, dataset_name, 'nnUNetPlans.json'))
dataset_json = load_json(join(nnUNet_preprocessed, dataset_name, 'dataset.json'))

trainer = nnUNetTrainerNyulIdentity(plans=plans, configuration='3d_fullres', fold=0,
                                    dataset_json=dataset_json,
                                    unpack_dataset=False, device=torch.device('cuda'))
trainer.initialize()  # builds wrapper, loads gamma table from _v2 cache

ckpt_path = Path(trainer.output_folder) / 'checkpoint_final.pth'
print(f'Loading {ckpt_path}', flush=True)
trainer.load_checkpoint(str(ckpt_path))
trainer.network.eval()
edain = trainer.network.edain        # the MRIEDAINLayer
backbone = trainer.network.backbone  # the UNet
gamma_table = trainer.network.case_gamma_table
print(f'gamma_table has {len(gamma_table)} cases', flush=True)

# ---------- Load each case's preprocessed b2nd, push the WHOLE volume through EDAIN ----------
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

pre_dir = Path(nnUNet_preprocessed) / dataset_name / 'nnUNetPlans_3d_fullres'
ds_cls = infer_dataset_class(str(pre_dir))

CASES = {
    'Lipo-001 (good, dice=0.955)': 'Lipo-001',
    'Lipo-021 (good, dice=0.983)': 'Lipo-021',
    'Lipo-077 (FAIL, dice=0.003)': 'Lipo-077',
    'Lipo-089 (FAIL, dice=0.000)': 'Lipo-089',
    'Lipo-097 (FAIL, dice=0.000)': 'Lipo-097',
}

# Pull all train cases too — we want training-distribution baseline
import json as _json
sp = _json.load(open(Path(nnUNet_preprocessed) / dataset_name / 'splits_final.json'))
train_ids = sp[0]['train']

# Sample 5 random train cases for distribution comparison
np.random.seed(0)
sample_train = list(np.random.choice(train_ids, size=5, replace=False))
for cid in sample_train:
    CASES[f'{cid} (train sample)'] = cid

ds = ds_cls(str(pre_dir), identifiers=list(set(CASES.values())))

def fwd_edain_whole_volume(cid):
    """Run EDAIN on the WHOLE preprocessed volume for case cid.
    Returns (z_in[fg], edain_out[fg]) on CPU as 1-D numpy."""
    data, _seg, _prev, _props = ds.load_case(cid)
    X_np = np.asarray(data[0])      # (D, H, W), already z-scored by nnU-Net
    X = torch.from_numpy(X_np).float().unsqueeze(0).unsqueeze(0).cuda()   # (1,1,D,H,W)
    trainer.network.set_current_batch([cid])
    with torch.no_grad():
        # Re-create the wrapper forward path WITHOUT the backbone
        mask = X != 0.0
        gamma_raw = trainer.network._lookup_gammas(1, X.device)
        x_out, _ = edain(X, mask=mask, gamma_raw=gamma_raw,
                         return_diagnostics=False)
    z_in_fg = X[mask].cpu().numpy()
    edain_fg = x_out[mask].cpu().numpy()
    return z_in_fg, edain_fg

# ---------- Diagnose ----------
print()
print('=' * 110)
print(f'{"CASE":<32} {"input fg (post-zscore)":<30}  {"EDAIN output fg":<30}')
print(f'{"":<32} {"mean   std   p05   p95":<30}  {"mean   std   p05   p95":<30}')
print('=' * 110)
for label, cid in CASES.items():
    z_in, z_out = fwd_edain_whole_volume(cid)
    print(f'{label:<32} '
          f'{z_in.mean():+6.2f} {z_in.std():5.2f} {np.percentile(z_in,5):+6.2f} {np.percentile(z_in,95):+6.2f}  '
          f'{z_out.mean():+6.2f} {z_out.std():5.2f} {np.percentile(z_out,5):+6.2f} {np.percentile(z_out,95):+6.2f}')

print()
print('INTERPRETATION:')
print('  - "input fg" = what nnU-Net z-score gives EDAIN (always ~mean=0, std=1)')
print('  - "EDAIN output fg" = what the UNet backbone actually sees')
print('  - If failing cases have EDAIN output stats wildly DIFFERENT from working/train cases,')
print('    the UNet is seeing out-of-distribution input -> diagnosis confirmed.')
print('  - If EDAIN output looks similar across all cases, the failure is INSIDE the UNet backbone.')
PY

echo "[diagnose] done $(date)"
