# HW2 – Linear Regression and Regression Trees from Scratch

Everything lives in the Jupyter notebook **`HW2.ipynb`**:

1. **Data generator**: `generate_noisy_dataset(feature_funcs, true_func, x_range, n_samples, noise, seed)` plus a fixed, seeded 80/20 `train_test_split`.
2. **Linear regression** (no ML library): `fit` (normal equation β = (XᵀX)⁻¹XᵀY), `predict`, `evaluate_model` (RMSE).
3. **Regression tree** (no ML library): class `RegressionTree` with `fit(X, y, min_records)`, `predict(X)`, `visualize(...)`.
4. **Experiments** for every report question (Q3–Q7 and Q9–Q11).

Generated outputs:
- `figures/`: every plot used in the report (PNG)
- `results/results.json`: every number quoted in the report

## Requirements

- Python 3.9 or newer (tested with Python 3.11)
- The packages listed in `requirements.txt` (`numpy`, `matplotlib`, `jupyter`, `nbconvert`)

## Build and run

```bash
# 1. (optional) create and activate a virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate

# 2. install the dependencies
pip install -r requirements.txt

# 3a. run interactively
jupyter notebook HW2.ipynb        # then: Kernel -> Restart & Run All

# 3b. or run headless (re-creates figures/ and results/ in place)
jupyter nbconvert --to notebook --execute --inplace HW2.ipynb
```

A full run takes about one minute. All randomness is seeded (`DATA_SEED = 42`, `SPLIT_SEED = 7`), so every run gives the same datasets, train/test splits, figures and numbers.
