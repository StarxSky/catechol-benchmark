from dataclasses import dataclass


@dataclass
class GNNConfig:
    # Architecture (from paper)
    hidden_dim: int = 256
    num_gat_layers: int = 4
    num_heads: int = 8
    drfp_dim: int = 2048

    # Training (from paper)
    lr: float = 3e-4
    weight_decay: float = 1e-5
    batch_size: int = 128
    max_epochs: int = 400
    dropout: float = 0.15
    head_dropout: float = 0.075
    grad_clip_norm: float = 1.0
    lr_plateau_factor: float = 0.7
    lr_plateau_patience: int = 30
    early_stop_patience: int = 50

    # Data
    device: str = "cuda"


@dataclass
class DataConfig:
    # Expected columns in the dataset CSV
    smiles_sm: str = "smiles_sm"
    smiles_p2: str = "smiles_p2"
    smiles_p3: str = "smiles_p3"
    smiles_solvent_a: str = "smiles_solvent_a"
    smiles_solvent_b: str = "smiles_solvent_b"  # may be empty for pure solvents

    solvent_a_id: str = "solvent_a_id"
    solvent_b_id: str = "solvent_b_id"
    percent_b: str = "percent_b"
    temperature: str = "temperature"
    residence_time: str = "residence_time"

    y_sm: str = "y_sm"
    y_p2: str = "y_p2"
    y_p3: str = "y_p3"

    # For LORO splits (mixture ramp id)
    ramp_id: str = "ramp_id"

    # Optional external DRFP features (n_samples x 2048)
    drfp_npy: str = ""
