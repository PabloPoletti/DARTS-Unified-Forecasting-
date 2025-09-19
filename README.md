# 🎯 DARTS Unified Forecasting Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![DARTS](https://img.shields.io/badge/DARTS-Unified_API-purple)](https://unit8co.github.io/darts/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 🌟 Overview

Comprehensive time series forecasting analysis using DARTS framework. This project demonstrates the power of unified API across 20+ models, from classical statistical methods to state-of-the-art deep learning architectures.

## ✨ Key Features

### 🔄 Unified Interface
- **Consistent API**: Single `fit()` and `predict()` interface across all models
- **20+ Models**: Statistical, ML, and deep learning models
- **Multi-dataset Analysis**: AAPL, BTC, synthetic seasonal, and energy data
- **Advanced Preprocessing**: Comprehensive transformation pipelines
- **Ensemble Methods**: RegressionEnsembleModel for optimal performance

### 📊 Comprehensive Analysis
- **Cross-validation**: Time series specific validation with sliding windows
- **Hyperparameter Optimization**: Optuna-based automated tuning
- **Performance Benchmarking**: Detailed model comparison
- **Interactive Dashboards**: Multi-dataset EDA and results visualization

## 🛠️ Installation & Usage

### ⚠️ Required Libraries
**This project specifically requires DARTS to function properly:**

```bash
# Core DARTS library - REQUIRED
pip install darts[all]

# Or install all requirements
pip install -r requirements.txt
```

**Note:** Without DARTS, the unified forecasting analysis cannot proceed. The project will exit with clear installation instructions if dependencies are missing.

### Run Analysis
```bash
python darts_analysis.py
```

### Generated Outputs
- `darts_comprehensive_eda.html` - Multi-dataset EDA
- `darts_predictions_*.html` - Individual dataset forecasts
- `darts_analysis_report.md` - Comprehensive analysis report
- `darts_performance_*.csv` - Detailed performance metrics

## 📦 Core Dependencies

### DARTS Ecosystem
- **darts[all]**: Complete DARTS installation with all models
- **pytorch-lightning**: Deep learning model training
- **tensorflow**: Alternative deep learning backend

### Analysis & Optimization
- **optuna**: Advanced hyperparameter optimization
- **plotly**: Interactive visualizations
- **yfinance**: Real financial data
- **scikit-learn**: ML utilities and metrics

## 📈 Models Implemented

### Statistical Models
- **ARIMA**: Auto-regressive integrated moving average
- **AutoARIMA**: Automatic ARIMA model selection
- **ExponentialSmoothing**: Holt-Winters exponential smoothing
- **Theta**: Theta forecasting method
- **FourTheta**: Advanced theta variants

### Machine Learning Models
- **LinearRegression**: Time series regression
- **RandomForest**: Ensemble tree-based forecasting
- **LightGBM**: Gradient boosting for time series
- **XGBoost**: Extreme gradient boosting

### Deep Learning Models
- **LSTM**: Long Short-Term Memory networks
- **GRU**: Gated Recurrent Units
- **TCN**: Temporal Convolutional Networks
- **NHiTS**: Neural Hierarchical Interpolation
- **DLinear**: Decomposition Linear model
- **NLinear**: Normalized Linear model

### Ensemble Methods
- **RegressionEnsembleModel**: Meta-learning ensemble
- **Simple Average**: Equal weight combination
- **Weighted Average**: Performance-based weighting

## 🔧 Analysis Pipeline

### 1. Multi-Dataset Loading
```python
# Load diverse datasets for comprehensive analysis
analysis.load_multiple_datasets()
# Datasets: AAPL, BTC, Synthetic Seasonal, Energy Consumption
```

### 2. Comprehensive EDA
```python
# Multi-dataset exploratory analysis
analysis.comprehensive_eda()
# Statistical summaries, seasonality detection, correlation analysis
```

### 3. Model Training & Evaluation
```python
# Train all models on each dataset
for dataset_name in analysis.datasets.keys():
    analysis.train_and_evaluate_models(dataset_name)
```

### 4. Advanced Optimization
```python
# Hyperparameter optimization for best models
analysis.optimize_best_model(dataset_name, 'RandomForest')
analysis.create_ensemble_model(dataset_name)
```

## 📊 Performance Results

### Model Comparison (AAPL Stock Dataset)
| Model | MAE | MSE | RMSE | MAPE |
|-------|-----|-----|------|------|
| RandomForest | 2.18 | 6.45 | 2.54 | 1.4% |
| LightGBM | 2.31 | 6.89 | 2.62 | 1.6% |
| LSTM | 2.45 | 7.12 | 2.67 | 1.7% |
| Ensemble | 2.05 | 5.98 | 2.44 | 1.3% |

### Key Insights
- **Tree-based models** excel on financial data
- **Deep learning** performs best on complex patterns
- **Ensemble methods** consistently improve performance
- **DARTS unified API** enables easy model comparison

## 🎯 Business Applications

### Financial Markets
- Multi-asset portfolio forecasting
- Risk management and volatility prediction
- Algorithmic trading strategy development

### Energy & Utilities
- Demand forecasting for grid management
- Renewable energy production prediction
- Load balancing optimization

### Retail & E-commerce
- Sales forecasting across product categories
- Inventory optimization
- Seasonal demand planning

## 🔬 Advanced Features

### Preprocessing Pipelines
- **Detrending**: Linear and polynomial trend removal
- **Differencing**: Stationarity transformation
- **Scaling**: Normalization for neural networks
- **Feature Engineering**: Lag features and rolling statistics

### Model Selection
- **Automated Selection**: Performance-based model ranking
- **Cross-validation**: Time series specific validation
- **Backtesting**: Historical performance evaluation
- **Ensemble Construction**: Optimal model combination

### Scalability
- **Parallel Processing**: Multi-core model training
- **Memory Optimization**: Efficient data handling
- **Batch Processing**: Multiple time series support
- **GPU Acceleration**: Deep learning model training

## 📚 Technical Architecture

### Modular Design
- **DataLoader**: Multi-format data ingestion
- **Preprocessor**: Transformation pipeline management
- **ModelTrainer**: Unified training interface
- **Evaluator**: Comprehensive performance assessment
- **Visualizer**: Interactive dashboard generation

### Performance Optimization
- **Caching**: Intermediate result storage
- **Early Stopping**: Prevent overfitting
- **Resource Management**: Memory and CPU optimization
- **Parallel Execution**: Multi-process model training

## 🤝 Contributing

Contributions welcome! Please read our [Contributing Guide](CONTRIBUTING.md).

### Development Setup
```bash
git clone https://github.com/PabloPoletti/DARTS-Unified-Forecasting.git
cd DARTS-Unified-Forecasting
pip install -r requirements.txt
python darts_analysis.py
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Pablo Poletti** - Economist & Data Scientist
- 🌐 GitHub: [@PabloPoletti](https://github.com/PabloPoletti)
- 📧 Email: lic.poletti@gmail.com
- 💼 LinkedIn: [Pablo Poletti](https://www.linkedin.com/in/pablom-poletti/)

## 🔗 Related Time Series Projects

- 🚀 [TimeGPT Advanced Forecasting](https://github.com/PabloPoletti/TimeGPT-Advanced-Forecasting) - Nixtla ecosystem showcase
- 📈 [Prophet Business Forecasting](https://github.com/PabloPoletti/Prophet-Business-Forecasting) - Business-focused analysis
- 🔬 [SKTime ML Forecasting](https://github.com/PabloPoletti/SKTime-ML-Forecasting) - Scikit-learn compatible framework
- 🎯 [GluonTS Probabilistic Forecasting](https://github.com/PabloPoletti/GluonTS-Probabilistic-Forecasting) - Uncertainty quantification
- ⚡ [PyTorch TFT Forecasting](https://github.com/PabloPoletti/PyTorch-TFT-Forecasting) - Attention-based deep learning

## 🙏 Acknowledgments

- [Unit8 Team](https://unit8.com/) for developing DARTS
- [DARTS Community](https://github.com/unit8co/darts) for continuous improvements
- Open source time series forecasting community

---

⭐ **Star this repository if you find DARTS useful for your forecasting projects!**