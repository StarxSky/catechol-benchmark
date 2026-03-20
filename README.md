# Learning Continuous Solvent Effects from Transient Flow Data

Reference implementation for the paper **“Learning Continuous Solvent Effects from Transient Flow Data: A Graph Neural Network Benchmark on Catechol Rearrangement”** (arXiv: 2512.19530v1).

This repository provides code for:
- GNN with GAT + DRFP + learned mixture encoding
- DeepModel (Transformer-enhanced SwiGLU MLP)
- GBDT baseline (multi-output)
- Ensemble (inverse-variance weighted)

## Contents
- `catechol_gnn/` core models and data utilities
- `train_gnn.py` training/evaluation for GNN
- `train_baselines.py` training/evaluation for GBDT/DeepModel/Ensemble
- `requirements.txt` dependencies

## Method Summary (Paper Alignment)
GNN implementation matches the paper’s design:
- 4 molecular graphs per sample: SM, P2, P3, Solvent
- GAT stack: 4 layers, 8 heads, hidden dim 256, residual connections
- Global mean + max pooling
- Learned mixture encoding: `e_mix = MLP([eA; eB; %B; T; time])`
- DRFP features: 2048-dim
- Final MLP head with sigmoid output for 3 yields
- Training: AdamW (lr `3e-4`, weight decay `1e-5`), batch `128`, max epochs `400`, early stopping `50`, dropout `0.15`, head dropout `0.075`, grad clip `1.0`, ReduceLROnPlateau (`factor=0.7`, `patience=30`)

DeepModel matches the paper’s design:
- Input projection to 384
- Single 8-head self-attention block
- 4 residual SwiGLU blocks
- 2-layer MLP output head
- Training: AdamW (lr `7e-4`, weight decay `1e-5`), batch `128`, max epochs `400`, early stopping `50`, dropout `0.15`, head dropout `0.075`, grad clip `1.0`

## Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Format
### Required CSV columns (GNN)
- `smiles_sm`, `smiles_p2`, `smiles_p3`
- `smiles_solvent_a`, `smiles_solvent_b` (empty for pure solvents)
- `solvent_a_id`, `solvent_b_id` (IDs used in LOSO)
- `percent_b`, `temperature`, `residence_time`
- `y_sm`, `y_p2`, `y_p3`
- `ramp_id` (for LORO)

Optional DRFP matrix: `.npy` with shape `(n_samples, 2048)`.

### Required CSV columns (Baselines)
The CSV must already include **numeric descriptor columns** (Spange, ACS PCA, DRFP), plus
`temperature`, `residence_time`, `percent_b`. All numeric columns except the 3 targets and
these 3 condition columns are treated as descriptor features.

## Usage
### GNN (LOSO / LORO)
```bash
python3 train_gnn.py --csv /path/to/catechol.csv --split loso --drfp /path/to/drfp.npy
python3 train_gnn.py --csv /path/to/catechol.csv --split loro --drfp /path/to/drfp.npy
```

### Baselines (GBDT / DeepModel / Ensemble)
```bash
python3 train_baselines.py --csv /path/to/catechol_tabular.csv --split loso
python3 train_baselines.py --csv /path/to/catechol_tabular.csv --split loro
```

## Reproducibility
- LOSO and LORO splits follow the paper’s protocol.
- GNN and DeepModel training follow the paper’s hyperparameters and early stopping criteria.
- DRFP features must be precomputed to reproduce reported metrics.

## Dependencies
- `torch`
- `torch-geometric`
- `rdkit`
- `numpy`, `pandas`
- `scikit-learn`

## License
This repository is released under the license specified by the authors. If you plan to distribute
or publish derived work, ensure you comply with the paper’s and dataset’s licenses.

## Citation
If you use this code, please cite the paper:
```
@article{xing2025catechol,
  title={Learning Continuous Solvent Effects from Transient Flow Data: A Graph Neural Network Benchmark on Catechol Rearrangement},
  author={Xing, Hongsheng and Si, Qiuxin},
  journal={arXiv preprint arXiv:2512.19530v1},
  year={2025}
}
```

## Contact
For questions about the dataset or methodology, please refer to the paper or contact the authors.
