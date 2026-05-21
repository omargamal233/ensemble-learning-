# Titanic Ensemble Model

A machine learning pipeline for Titanic survival prediction using ensemble methods.

## Quick Start

```bash
pip install -r requirements.txt
python main.py --data-dir data/raw --output-dir outputs
```

## Project Structure

```
titanic-ensemble/
├── main.py                 # Entry point
├── requirements.txt        # Dependencies
├── README.md              # This file
├── data/
│   └── raw/
│       ├── train.csv      # Training data
│       └── test.csv       # Test data
├── outputs/               # Generated results
│   ├── submission_voting.csv
│   └── figures/          # Visualization plots (22 PNG files)
└── src/
    ├── __init__.py
    ├── config.py          # Configuration & paths
    ├── main.py            # Pipeline orchestration
    ├── data/              # Data loading & preprocessing
    │   ├── load.py
    │   └── preprocess.py
    ├── features/          # Feature engineering
    │   └── engineer.py
    └── models/            # Model training & evaluation
        ├── train.py
        ├── ensemble.py
        └── evaluate.py
```

## What it Does

1. **Loads & Preprocesses** Titanic data (handles missing values, outliers, feature encoding)
2. **Trains 5 Base Models**: Random Forest, Extra Trees, AdaBoost, Gradient Boosting, SVC
3. **Builds 2 Ensemble Methods**: Voting (soft voting) and Stacking (LogisticRegression meta-learner)
4. **Evaluates All Models** with cross-validation and comprehensive metrics
5. **Generates 22 Visualizations**: Confusion matrices, ROC curves, probability distributions
6. **Creates Predictions** for Kaggle submission using the best ensemble

## Usage

```bash
# Default run
python main.py

# Custom data directory
python main.py --data-dir data/raw

# Custom output directory
python main.py --output-dir results

# Skip outlier removal
python main.py --skip-outliers

# Custom CV folds
python main.py --cv 10
```

## Results

The pipeline outputs:
- **submission_voting.csv** - Final predictions for Kaggle
- **figures/** - 22 PNG visualizations (3 per base model + ensemble plots)
- **Python results dict** containing:
  - Base model metrics and evaluation data
  - Ensemble model metrics and evaluation data
  - All matplotlib figures
  - File paths to saved visualizations
  - Predictions and passenger IDs

## Best Model

Voting Ensemble typically achieves **~82.6%** accuracy on cross-validation.

