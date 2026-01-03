# Water Surface Classification in Kazakhstan Based on Satellite Imagery

## Assignment 1: Machine Learning for Remote Sensing

This project implements comprehensive machine learning methods for classifying water surface images in Kazakhstan using satellite imagery data. The work includes 11 different regression algorithms evaluated through 10-fold cross-validation.

---

## Project Structure

```
Assignment 1/
├── .venv/                  # Virtual environment (create with: python -m venv .venv)
├── data/                   # Dataset storage
│   └── water_surface_dataset.csv
├── src/                    # Source code
│   ├── preprocessing.py    # Data preprocessing utilities
│   ├── features.py         # Feature engineering (NDWI, MNDWI, etc.)
│   ├── models.py           # ML model implementations
│   ├── evaluation.py       # Model evaluation and metrics
│   └── main.py            # Main execution script
├── results/                # Results and outputs
│   └── table1_results.csv # Table 1: Model Performance Summary
├── figures/                # Generated visualizations
│   ├── model_comparison.png
│   ├── cv_performance.png
│   └── feature_importance.png
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── academic_article.md    # Full academic research article
└── technical_report.md    # Technical report with methodology
```

---

## Dataset

**Dataset Name:** Synthetic Sentinel-2 Multispectral Imagery Dataset for Water Surface Classification

**Description:** 
This dataset simulates Sentinel-2 satellite imagery with multispectral bands (Blue, Green, Red, NIR, SWIR1, SWIR2) and derived water indices (NDWI, MNDWI, NDVI, EVI). The dataset contains 5,000 samples representing water and non-water surfaces, suitable for regression-based water surface probability prediction.

**Features:**
- Spectral bands: B2_Blue, B3_Green, B4_Red, B8_NIR, B11_SWIR1, B12_SWIR2
- Water indices: NDWI, MNDWI, NDVI, EVI
- Band ratios: NIR_Red_Ratio, SWIR_NIR_Ratio

**Target:** Water_Surface_Probability (continuous value 0-1)

**Relevance to Kazakhstan:**
The dataset structure mirrors Sentinel-2 data available for Kazakhstan through the Copernicus Open Access Hub. Kazakhstan's diverse water bodies (rivers, lakes, reservoirs) can be effectively analyzed using these spectral characteristics. The synthetic data generation follows established patterns observed in Central Asian water bodies.

**Access:**
The dataset is automatically generated on first run and saved to `data/water_surface_dataset.csv`. For real Sentinel-2 data, access via:
- [Copernicus Open Access Hub](https://scihub.copernicus.eu/)
- [USGS Earth Explorer](https://earthexplorer.usgs.gov/)

---

## Setup Instructions

### 1. Create Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python -c "import sklearn, xgboost, lightgbm, catboost; print('All packages installed successfully')"
```

---

## Execution Instructions

### Run Complete Pipeline

```bash
python src/main.py
```

This will:
1. Generate/load the dataset
2. Perform feature engineering
3. Preprocess the data
4. Train and evaluate all 11 models with 10-fold cross-validation
5. Generate Table 1 results
6. Create visualizations

### Expected Output

The script will:
- Display progress for each step
- Show evaluation metrics for each model
- Save results to `results/table1_results.csv`
- Generate figures in `figures/` directory
- Print Table 1 summary

### Execution Time

Approximate execution time: 5-15 minutes (depending on hardware)

---

## Implemented Algorithms

All algorithms are implemented as regression models:

1. **Ridge Regression** - L2 regularization
2. **Lasso Regression** - L1 regularization
3. **Elastic Net** - Combined L1/L2 regularization
4. **k-Nearest Neighbors (KNN) Regression** - Instance-based learning
5. **Extra Trees Regression** - Extremely randomized trees
6. **AdaBoost Regression** - Adaptive boosting
7. **Gradient Boosting Regression** - Sequential boosting
8. **XGBoost** - Extreme gradient boosting
9. **LightGBM** - Light gradient boosting machine
10. **CatBoost** - Categorical boosting
11. **HistGradientBoosting** - Histogram-based gradient boosting

---

## Evaluation Methodology

- **Cross-Validation:** 10-fold (k=10)
- **Metrics:** RMSE (Root Mean Square Error) and R² (Coefficient of Determination)
- **Preprocessing:** StandardScaler for feature normalization
- **Reproducibility:** Fixed random seed (42)

---

## Results

### Table 1: Model Performance Summary

Results are saved to `results/table1_results.csv` and include:
- Algorithm name
- Number of features
- Number of targets
- k-fold validation (k=10)
- RMSE (mean ± std)
- R² score (mean ± std)

### Visualizations

Three types of visualizations are generated:

1. **Model Comparison** (`figures/model_comparison.png`)
   - Bar charts comparing RMSE and R² across all models

2. **Cross-Validation Performance** (`figures/cv_performance.png`)
   - Box plots showing distribution of metrics across 10 folds

3. **Feature Importance** (`figures/feature_importance.png`)
   - Top 10 most important features for tree-based models

---

## Code Organization

### `src/preprocessing.py`
- `DataPreprocessor`: Handles data loading, scaling, and train/test splitting

### `src/features.py`
- `FeatureEngineer`: Calculates spectral indices (NDWI, MNDWI, NDVI, EVI)

### `src/models.py`
- `ModelFactory`: Creates and configures all ML models

### `src/evaluation.py`
- `ModelEvaluator`: Performs cross-validation and metric calculation

### `src/main.py`
- Main orchestration script that runs the complete pipeline

---

## Reproducibility

To ensure reproducibility:
- Random seed is set to 42 throughout
- All models use consistent random_state parameters
- Cross-validation uses fixed shuffling with random_state=42

---

## Requirements

- Python 3.8+
- See `requirements.txt` for full dependency list

---

## Author

Baibossyn Aniyar

---

## License

This work is produced for academic assignment purposes.

---

## References

See `academic_article.md` and `technical_report.md` for complete references and citations.

