"""
🎯 DARTS Unified Forecasting Dashboard
Professional Time Series Forecasting with Unified Interface

Author: Pablo Poletti
GitHub: https://github.com/PabloPoletti
Contact: lic.poletti@gmail.com
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

# DARTS imports
try:
    from darts import TimeSeries
    from darts.models import (
        # Statistical models
        ARIMA, AutoARIMA, ExponentialSmoothing, Theta, Prophet,
        LinearRegressionModel, RandomForest, LightGBMModel, XGBModel,
        # Deep learning models  
        RNNModel, LSTMModel, GRUModel, TCNModel, TFTModel,
        NBEATSModel, NHiTSModel, TransformerModel, DLinearModel,
        # Ensemble models
        RegressionEnsembleModel, NaiveEnsembleModel
    )
    from darts.metrics import mape, mae, mse, rmse, smape
    from darts.utils.statistics import check_seasonality, extract_trend_and_seasonality
    from darts.dataprocessing.transformers import Scaler, StaticCovariatesTransformer
    from darts.utils.timeseries_generation import datetime_attribute_timeseries
except ImportError as e:
    st.error(f"Error importing DARTS: {e}")
    st.stop()

# ML and optimization
from sklearn.model_selection import TimeSeriesSplit
import optuna

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger('lightning.pytorch').setLevel(logging.WARNING)

# Page config
st.set_page_config(
    page_title="DARTS Unified Forecasting",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .model-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        text-align: center;
    }
    .performance-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Constants
COLORS = {
    'primary': '#667eea',
    'secondary': '#764ba2', 
    'accent': '#f093fb',
    'success': '#96CEB4',
    'warning': '#FECA57',
    'error': '#FF9FF3'
}

@st.cache_data
def generate_sample_data() -> Dict[str, TimeSeries]:
    """Generate sample time series data using DARTS"""
    
    datasets = {}
    
    # 1. Retail sales with multiple seasonalities
    dates = pd.date_range('2020-01-01', '2024-12-01', freq='D')
    
    # Create complex seasonal pattern
    trend = np.linspace(1000, 2500, len(dates))
    weekly = 200 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
    monthly = 150 * np.sin(2 * np.pi * np.arange(len(dates)) / 30.44)
    yearly = 300 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    
    # Add holidays effect
    holiday_dates = ['2020-12-25', '2021-12-25', '2022-12-25', '2023-12-25', '2024-12-25']
    holiday_effect = np.zeros(len(dates))
    for holiday in holiday_dates:
        if holiday in dates.strftime('%Y-%m-%d'):
            idx = np.where(dates.strftime('%Y-%m-%d') == holiday)[0][0]
            # 3-day holiday effect
            for i in range(max(0, idx-1), min(len(dates), idx+2)):
                holiday_effect[i] = 500
    
    noise = np.random.normal(0, 50, len(dates))
    sales = trend + weekly + monthly + yearly + holiday_effect + noise
    sales = np.maximum(sales, 100)
    
    datasets['Retail_Sales'] = TimeSeries.from_times_and_values(
        times=dates,
        values=sales,
        freq='D'
    )
    
    # 2. Energy consumption with complex patterns
    dates_hourly = pd.date_range('2023-01-01', '2024-06-01', freq='H')
    
    # Daily, weekly, and seasonal patterns
    hour_effect = 30 * np.sin(2 * np.pi * dates_hourly.hour / 24)
    day_effect = -10 * (dates_hourly.weekday >= 5).astype(int)  # Weekend effect
    seasonal_effect = 20 * np.sin(2 * np.pi * dates_hourly.dayofyear / 365.25)
    
    base_consumption = 100
    noise_hourly = np.random.normal(0, 5, len(dates_hourly))
    energy = base_consumption + hour_effect + day_effect + seasonal_effect + noise_hourly
    energy = np.maximum(energy, 20)
    
    # Sample for performance
    sample_idx = np.arange(0, len(energy), 6)  # Every 6 hours
    
    datasets['Energy_Consumption'] = TimeSeries.from_times_and_values(
        times=dates_hourly[sample_idx],
        values=energy[sample_idx],
        freq='6H'
    )
    
    # 3. Financial time series (stock-like)
    dates_business = pd.date_range('2020-01-01', '2024-12-01', freq='B')  # Business days
    
    # Random walk with drift and volatility clustering
    returns = np.random.normal(0.0005, 0.02, len(dates_business))  # Daily returns
    # Add volatility clustering
    for i in range(1, len(returns)):
        if abs(returns[i-1]) > 0.03:  # High volatility day
            returns[i] *= 1.5  # Increase volatility
    
    price = 100 * np.exp(np.cumsum(returns))
    
    datasets['Stock_Price'] = TimeSeries.from_times_and_values(
        times=dates_business,
        values=price,
        freq='B'
    )
    
    return datasets

def get_model_categories() -> Dict[str, List]:
    """Get categorized models available in DARTS"""
    
    return {
        'Statistical': [
            ('ARIMA', ARIMA),
            ('AutoARIMA', AutoARIMA),
            ('Exponential Smoothing', ExponentialSmoothing),
            ('Theta', Theta),
            ('Prophet', Prophet)
        ],
        'Machine Learning': [
            ('Linear Regression', LinearRegressionModel),
            ('Random Forest', RandomForest),
            ('LightGBM', LightGBMModel),
            ('XGBoost', XGBModel)
        ],
        'Deep Learning': [
            ('LSTM', LSTMModel),
            ('GRU', GRUModel),
            ('TCN', TCNModel),
            ('TFT', TFTModel),
            ('NBEATS', NBEATSModel),
            ('NHiTS', NHiTSModel),
            ('Transformer', TransformerModel),
            ('DLinear', DLinearModel)
        ]
    }

def create_model_instance(model_name: str, model_class, **kwargs):
    """Create model instance with appropriate parameters"""
    
    # Default parameters for different model types
    if model_name in ['LSTM', 'GRU', 'RNN']:
        return model_class(
            input_chunk_length=12,
            training_length=20,
            n_epochs=10,
            **kwargs
        )
    elif model_name in ['TCN', 'TFT', 'NBEATS', 'NHiTS', 'Transformer', 'DLinear']:
        return model_class(
            input_chunk_length=12,
            output_chunk_length=6,
            n_epochs=10,
            **kwargs
        )
    elif model_name == 'Prophet':
        return model_class(**kwargs)
    elif model_name in ['AutoARIMA', 'ARIMA']:
        return model_class(**kwargs)
    elif model_name in ['Random Forest', 'LightGBM', 'XGBoost']:
        return model_class(
            lags=12,
            **kwargs
        )
    else:
        return model_class(**kwargs)

def optimize_model_hyperparameters(series: TimeSeries, model_name: str, model_class) -> Dict:
    """Optimize hyperparameters using Optuna"""
    
    def objective(trial):
        try:
            if model_name == 'Random Forest':
                params = {
                    'lags': trial.suggest_int('lags', 6, 24),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 15)
                }
            elif model_name == 'LSTM':
                params = {
                    'input_chunk_length': trial.suggest_int('input_chunk_length', 6, 24),
                    'hidden_dim': trial.suggest_categorical('hidden_dim', [16, 32, 64]),
                    'n_rnn_layers': trial.suggest_int('n_rnn_layers', 1, 3),
                    'training_length': 20,
                    'n_epochs': 5
                }
            elif model_name == 'TCN':
                params = {
                    'input_chunk_length': trial.suggest_int('input_chunk_length', 6, 24),
                    'output_chunk_length': trial.suggest_int('output_chunk_length', 1, 12),
                    'kernel_size': trial.suggest_int('kernel_size', 2, 5),
                    'num_filters': trial.suggest_categorical('num_filters', [16, 32, 64]),
                    'n_epochs': 5
                }
            else:
                return float('inf')
            
            # Split data for validation
            train, val = series.split_before(0.8)
            
            # Create and train model
            model = create_model_instance(model_name, model_class, **params)
            model.fit(train)
            
            # Make prediction
            pred = model.predict(len(val))
            error = mae(val, pred)
            
            return error
            
        except Exception:
            return float('inf')
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10, show_progress_bar=False)
    
    return study.best_params

def perform_model_comparison(series: TimeSeries, models: Dict[str, List], horizon: int) -> pd.DataFrame:
    """Compare multiple models using time series cross-validation"""
    
    results = []
    
    # Split data
    train_size = int(len(series) * 0.8)
    train_series = series[:train_size]
    test_series = series[train_size:]
    
    for category, model_list in models.items():
        for model_name, model_class in model_list:
            try:
                # Create model instance
                model = create_model_instance(model_name, model_class)
                
                # Fit model
                model.fit(train_series)
                
                # Make prediction
                pred = model.predict(horizon)
                
                # Calculate metrics
                actual_values = test_series[:horizon] if len(test_series) >= horizon else test_series
                pred_values = pred[:len(actual_values)]
                
                mae_score = mae(actual_values, pred_values)
                mse_score = mse(actual_values, pred_values)
                mape_score = mape(actual_values, pred_values)
                
                results.append({
                    'Category': category,
                    'Model': model_name,
                    'MAE': mae_score,
                    'MSE': mse_score,
                    'MAPE': mape_score
                })
                
            except Exception as e:
                st.warning(f"Error with {model_name}: {str(e)}")
                continue
    
    return pd.DataFrame(results)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🎯 DARTS Unified Forecasting</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="performance-card">
    🎯 <strong>Professional Time Series Forecasting with DARTS Framework</strong><br>
    Unified interface for classical, ML, and deep learning models with consistent fit() and predict() API
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎛️ Model Configuration")
        
        # Dataset selection
        datasets = generate_sample_data()
        dataset_name = st.selectbox("📊 Select Dataset:", list(datasets.keys()))
        series = datasets[dataset_name]
        
        # Model selection
        st.markdown("### 🤖 Model Selection")
        model_categories = get_model_categories()
        
        selected_models = {}
        for category, models in model_categories.items():
            st.markdown(f"**{category} Models:**")
            selected_models[category] = []
            for model_name, model_class in models:
                if st.checkbox(f"{model_name}", key=f"{category}_{model_name}"):
                    selected_models[category].append((model_name, model_class))
        
        # Forecasting parameters
        st.markdown("### 📈 Forecasting Setup")
        forecast_horizon = st.slider("🔮 Forecast Horizon:", 6, 48, 12)
        
        # Advanced options
        st.markdown("### ⚙️ Advanced Options")
        enable_optimization = st.checkbox("🔧 Hyperparameter Optimization", value=False)
        enable_ensemble = st.checkbox("🎭 Ensemble Methods", value=False)
        show_residuals = st.checkbox("📊 Show Residuals Analysis", value=False)
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Data visualization
        st.subheader("📊 Time Series Analysis")
        
        # Convert to pandas for plotting
        df = series.pd_dataframe()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df.iloc[:, 0],
            mode='lines',
            name='Time Series',
            line=dict(color=COLORS['primary'], width=2)
        ))
        
        fig.update_layout(
            title=f"Dataset: {dataset_name}",
            xaxis_title="Date",
            yaxis_title="Value",
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Time series statistics
        st.markdown("### 📈 Time Series Properties")
        
        col_a, col_b, col_c, col_d = st.columns(4)
        
        with col_a:
            st.metric("📊 Length", len(series))
        with col_b:
            st.metric("📅 Frequency", series.freq)
        with col_c:
            mean_val = float(series.mean().values()[0])
            st.metric("📈 Mean", f"{mean_val:.2f}")
        with col_d:
            std_val = float(series.std().values()[0])
            st.metric("📊 Std Dev", f"{std_val:.2f}")
        
        # Seasonality analysis
        is_seasonal, period = check_seasonality(series)
        if is_seasonal:
            st.success(f"🔄 Seasonality detected with period: {period}")
        else:
            st.info("🔄 No clear seasonality detected")
    
    with col2:
        # Model performance summary
        st.subheader("🏆 Model Performance")
        
        # Check if any models are selected
        total_selected = sum(len(models) for models in selected_models.values())
        
        if total_selected > 0:
            st.info(f"📊 {total_selected} models selected")
            
            # Show selected models
            for category, models in selected_models.items():
                if models:
                    st.markdown(f"**{category}:**")
                    for model_name, _ in models:
                        st.markdown(f"• {model_name}")
        else:
            st.warning("⚠️ Please select at least one model")
    
    # Model comparison and forecasting
    if total_selected > 0 and st.button("🚀 Run Forecasting Analysis", type="primary"):
        st.markdown("---")
        st.subheader("🔮 Forecasting Results")
        
        with st.spinner("Training models and generating forecasts..."):
            # Perform model comparison
            comparison_results = perform_model_comparison(
                series, selected_models, forecast_horizon
            )
            
            if not comparison_results.empty:
                # Display comparison table
                st.markdown("### 📊 Model Comparison")
                
                # Sort by MAE
                comparison_results = comparison_results.sort_values('MAE')
                st.dataframe(comparison_results, hide_index=True)
                
                # Best model visualization
                best_model_info = comparison_results.iloc[0]
                st.success(f"🏆 Best Model: {best_model_info['Model']} (MAE: {best_model_info['MAE']:.3f})")
                
                # Generate forecasts with best models
                st.markdown("### 🔮 Forecast Visualization")
                
                # Split data for visualization
                train_size = int(len(series) * 0.8)
                train_series = series[:train_size]
                test_series = series[train_size:]
                
                fig = go.Figure()
                
                # Historical data
                train_df = train_series.pd_dataframe()
                test_df = test_series.pd_dataframe()
                
                fig.add_trace(go.Scatter(
                    x=train_df.index,
                    y=train_df.iloc[:, 0],
                    mode='lines',
                    name='Training Data',
                    line=dict(color=COLORS['primary'], width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=test_df.index,
                    y=test_df.iloc[:, 0],
                    mode='lines',
                    name='Actual',
                    line=dict(color=COLORS['secondary'], width=2)
                ))
                
                # Generate forecasts for top 3 models
                colors = [COLORS['accent'], COLORS['success'], COLORS['warning']]
                
                for i, (_, row) in enumerate(comparison_results.head(3).iterrows()):
                    try:
                        # Find the model class
                        model_class = None
                        for category, models in selected_models.items():
                            for model_name, m_class in models:
                                if model_name == row['Model']:
                                    model_class = m_class
                                    break
                        
                        if model_class:
                            # Create and train model
                            model = create_model_instance(row['Model'], model_class)
                            model.fit(train_series)
                            
                            # Generate forecast
                            forecast = model.predict(forecast_horizon)
                            forecast_df = forecast.pd_dataframe()
                            
                            fig.add_trace(go.Scatter(
                                x=forecast_df.index,
                                y=forecast_df.iloc[:, 0],
                                mode='lines+markers',
                                name=f"{row['Model']} Forecast",
                                line=dict(color=colors[i % len(colors)], width=2, dash='dash')
                            ))
                    
                    except Exception as e:
                        st.warning(f"Error generating forecast for {row['Model']}: {str(e)}")
                        continue
                
                fig.update_layout(
                    title="📈 Time Series Forecasting Comparison",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    template="plotly_white",
                    height=600,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Download results
                csv = comparison_results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name=f"darts_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
    🎯 <strong>DARTS Unified Forecasting</strong> | 
    Built with DARTS framework | 
    <a href="https://github.com/PabloPoletti" target="_blank">GitHub</a> | 
    <a href="mailto:lic.poletti@gmail.com">Contact</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
