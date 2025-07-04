"""
CCTeu Pricing Model Package
===========================

A comprehensive pricing model for Italian CCTeu (Certificati di Credito del Tesoro 
indicizzati all'Euribor) bonds with advanced quantitative analytics.

Modules:
--------
- bloomberg_api: Bloomberg Terminal data interface
- preprocessing: Data cleaning and transformation utilities
- features: Feature engineering for pricing models
- models: Machine learning and statistical models
- relative_value: Cross-sectional analysis and relative value signals

Author: Quantitative Analytics Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Quantitative Analytics Team"

# Core imports for easy access
from .bloomberg_api import BloombergAPI
from .preprocessing import (
    calculate_daily_returns,
    calculate_log_returns,
    rolling_zscore,
    calculate_spread
)
from .features import build_feature_set
from .models import train_model
from .relative_value import (
    compute_pca_components,
    compute_cluster_index,
    compute_residual_series,
    compute_relative_value_signals
)

__all__ = [
    'BloombergAPI',
    'calculate_daily_returns',
    'calculate_log_returns',
    'rolling_zscore',
    'calculate_spread',
    'build_feature_set',
    'train_model',
    'compute_pca_components',
    'compute_cluster_index',
    'compute_residual_series',
    'compute_relative_value_signals'
]
