# Baseline Feature Expectations

The paper's tabular baselines (GBDT, DeepModel, Ensemble) use a **concatenation of**:
- Spange solvent descriptors
- ACS PCA descriptors
- DRFP descriptors
- Numeric conditions: temperature, residence time, %B

In `train_baselines.py`, the CSV is expected to already include these numeric descriptors.
All numeric columns except the 3 targets (`y_sm`, `y_p2`, `y_p3`) and the three condition columns
(`temperature`, `residence_time`, `percent_b`) are treated as descriptors.
