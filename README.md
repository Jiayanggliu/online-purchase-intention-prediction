# Online Shopper Purchase Intention Prediction

> Can browsing behavior predict whether an e-commerce session will end in a purchase?

A machine-learning classification project built on the **UCI Online Shoppers Purchasing Intention** dataset. The project compares Logistic Regression, Random Forest, and a soft-voting ensemble, then turns the strongest model into an interactive Dash experience for exploring conversion behavior and estimating purchase probability.

## Project Snapshot

| Metric | Result |
|---|---:|
| Sessions after duplicate removal | 12,205 |
| Overall conversion rate | 15.63% |
| Best single model | Random Forest |
| Random Forest accuracy | 89.6% |
| Random Forest F1 | 0.688 |
| Purchase recall | 73.0% |
| Purchase precision | 65.0% |

Because purchases are the minority class, the analysis emphasizes **F1, recall, precision, and confusion-matrix behavior** rather than accuracy alone.

## What the Project Does

1. Loads and validates 12K+ e-commerce sessions.
2. Removes exact duplicates and converts the purchase target to binary form.
3. Applies log transforms to skewed continuous variables.
4. One-hot encodes categorical browsing/session attributes.
5. Removes highly correlated features to reduce redundancy.
6. Trains and tunes Logistic Regression and Random Forest classifiers.
7. Compares both models with a soft-voting ensemble.
8. Tunes decision thresholds using precision-recall tradeoffs.
9. Builds business-facing KPI, conversion, feature-importance, and prediction views in Dash.

## Model Comparison

| Model | Accuracy | F1 | Recall | Precision |
|---|---:|---:|---:|---:|
| Logistic Regression* | 0.883 | 0.677 | 0.785 | 0.595 |
| Random Forest | **0.896** | **0.688** | **0.730** | 0.650 |
| Ensemble | 0.898 | 0.675 | 0.673 | **0.676** |

\*The original team notebook evaluated Logistic Regression on an unscaled test matrix in one cell even though the model was trained on scaled features. `portfolio_analysis.ipynb` corrects that line and clears stale notebook outputs so the corrected version can be rerun cleanly. The Random Forest metrics shown above are unaffected by that issue.

## Business Takeaway

Random Forest offers the best F1 balance for the imbalanced purchase target. At the selected threshold, it identifies roughly **73% of actual purchases** while maintaining about **65% precision**. The dashboard makes the model easier to interpret by pairing prediction with conversion KPIs, monthly patterns, visitor-type behavior, and feature importance.

## Repository Structure

```text
online-purchase-intention-prediction/
├── app/
│   └── app.py
├── assets/
├── data/
│   ├── README.md
│   └── online_shoppers_intention.csv
├── notebooks/
│   ├── original_team_notebook.ipynb
│   └── portfolio_analysis.ipynb
├── .gitignore
├── README.md
├── ROADMAP.md
└── requirements.txt
```

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python app/app.py
```

Then open the local Dash URL shown in the terminal.

## Dataset

**UCI Machine Learning Repository — Online Shoppers Purchasing Intention Dataset**  
https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

The target variable is `Revenue`: `True` means the session ended in a purchase.

## Team Attribution

Original course project by **Nisha Thiagaraj, Jiayang Liu, Ruoyu Yan, and Bo Xu**. This repository is a portfolio-oriented restructuring of the team notebook and retains the original team attribution.

## Next Improvements

See [`ROADMAP.md`](ROADMAP.md) for planned extensions including model calibration, SHAP explanations, deployment, and a more product-like prediction interface.
