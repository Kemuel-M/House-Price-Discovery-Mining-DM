# 🏠 House Price Discovery & Mining (DM)

A robust, end-to-end Data Science pipeline for predicting house prices and discovering hidden patterns in real estate data. This project goes beyond simple regression by incorporating **Frequent Pattern Mining** and **Segmented Error Analysis** to provide actionable business insights.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Project Overview

This repository implements a modular pipeline to solve the classic Ames Housing dataset problem. The goal is twofold:
1.  **Predictive Modeling**: Achieve high accuracy in house price estimation using XGBoost.
2.  **Insight Discovery**: Use Association Rules and Sequential Patterns to understand what combinations of features drive high-value sales.

### Key Results
- **R² Score**: ~0.906 (10-fold Cross-Validation)
- **RMSE (log)**: ~0.117
- **Mean Absolute Error (MAE)**: ~$12,500 on original scale
- **Primary Price Drivers**: `OverallQual`, `QualAreaInteract` (Quality * Area), and `Neighborhood`.

---

## 🚀 Key Features

### 🛠️ Advanced Data Engineering
- **Robust Imputation**: Neighborhood-based median imputation for `LotFrontage` and logical fill-ins for categorical features (e.g., 'NA' for houses without garages).
- **Multivariate Outlier Detection**: Combines domain-specific rules with **Isolation Forest** to ensure high-quality training data.
- **Smart Feature Engineering**: 
    - `QualAreaInteract`: Captures the non-linear relationship between quality and living area.
    - `NbPriceTier`: Groups 25+ neighborhoods into 5 distinct price levels to reduce dimensionality while preserving signal.
    - `TotalBathrooms` & `HouseAge`: Consolidated metrics for better model performance.

### 🔍 Discovery Mining (DM)
- **Association Rules (Apriori)**: Identifies feature combinations that lead to high-value "Muito Alto" price segments (e.g., *GarageCars=Alto + High Quality -> SalePrice=Muito Alto*).
- **Sequential Patterns (PrefixSpan)**: Analyzes the "evolutionary patterns" of house attributes across different construction decades.
- **Market Segmentation**: Uses **K-Means Clustering** to segment the market into distinct clusters (e.g., entry-level, luxury, renovated old homes).

### 📈 Detailed Error Analysis
- The pipeline generates a **Decile Error Report**, showing that the model is most accurate for mid-range homes but faces challenges in the lowest 10% (D1) price segment, allowing for targeted future improvements.

---

## 📁 Project Structure

```text
├── data/
│   ├── raw/                # Original train/test files
│   └── processed/          # Final submission.csv
├── notebooks/              # Experimental EDA and Mining (Jupyter)
├── reports/
│   ├── figures/            # Exported charts (Correlation, MI, Features)
│   └── execution_log_*.txt # Detailed logs of every pipeline run
├── src/
│   ├── cleaning.py         # Imputation, Outliers, Feature Engineering
│   ├── mining.py           # Apriori, PrefixSpan, Clustering
│   ├── models.py           # XGBoost training and evaluation
│   └── utils.py            # Visualization and helper functions
├── main.py                 # Main orchestrator (Entry point)
└── requirements.txt        # Dependencies
```

---

## 🛠️ Installation & Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Kemuel-M/House-Price-Discovery-Mining-DM.git
   cd House-Price-Discovery-Mining-DM
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the full pipeline**:
   ```bash
   python main.py
   ```
   *This will execute the entire process: cleaning, EDA, mining, training, and generating a Kaggle-ready `submission.csv`.*

---

## 📊 Visualizations

The pipeline automatically generates several reports in `reports/figures/`:
- **`informacao_mutua.png`**: Top features ranked by Mutual Information.
- **`heatmap_correlacao.png`**: Pearson vs. Spearman correlation matrix.
- **`importancia_features_model.png`**: Global importance assigned by XGBoost.
- **`resultados_regressao.png`**: Residual plots and predicted vs. real values.

---

## ✉️ Contact
Created by [Kemuel Marvila](https://github.com/Kemuel-M/) - feel free to reach out for collaborations or inquiries!
