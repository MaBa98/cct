"""
Main execution script for CCTeu Pricing Model
============================================

This script orchestrates the entire pricing workflow:
1. Data extraction from Bloomberg Terminal
2. Feature engineering and preprocessing
3. Model training and validation
4. Relative value analysis
5. Results output and visualization

Usage:
    python main.py [--config config/api_keys.yaml] [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD]
"""

import argparse
import logging
import yaml
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

from src import (
    BloombergAPI,
    build_feature_set,
    train_model,
    compute_relative_value_signals
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ccteu_pricing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_ccteu_universe() -> list:
    """
    Define the universe of CCTeu bonds to analyze.
    Returns list of Bloomberg tickers.
    """
    return [
        'IT0003644870 Corp',  # CCTeu Dic23
        'IT0003765865 Corp',  # CCTeu Mag24
        'IT0003934657 Corp',  # CCTeu Set24
        'IT0004019581 Corp',  # CCTeu Gen25
        'IT0004164528 Corp',  # CCTeu Lug25
        'IT0004273914 Corp',  # CCTeu Gen26
        'IT0004423428 Corp',  # CCTeu Lug26
        'IT0004513343 Corp',  # CCTeu Gen27
        'IT0004634917 Corp',  # CCTeu Lug27
        'IT0004756357 Corp',  # CCTeu Gen28
    ]


def get_benchmark_universe() -> list:
    """Define benchmark instruments for model drivers."""
    return [
        'RX1 Comdty',      # Bund Future
        'IK1 Comdty',      # BTP Future
        'EUR006M Index',   # Euribor 6M
        'EONIA Index',     # EONIA
        'EUSWE5 Index',    # EUR 5Y Swap
        'EUSWE10 Index',   # EUR 10Y Swap
    ]


def extract_data(api: BloombergAPI, tickers: list, start_date: datetime, end_date: datetime) -> dict:
    """Extract historical price data from Bloomberg."""
    logger.info(f"Extracting data for {len(tickers)} instruments from {start_date} to {end_date}")
    
    fields = ['PX_LAST', 'PX_OPEN', 'PX_HIGH', 'PX_LOW', 'PX_VOLUME']
    
    try:
        data = api.get_historical_data(tickers, fields, start_date, end_date)
        logger.info(f"Successfully extracted data for {len(data)} instruments")
        return data
    except Exception as e:
        logger.error(f"Error extracting data: {str(e)}")
        raise


def run_pricing_model(config: dict, start_date: datetime, end_date: datetime):
    """Main pricing model execution."""
    
    # Initialize data directory
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    Path('logs').mkdir(exist_ok=True)
    
    # Get instrument universe
    ccteu_tickers = get_ccteu_universe()
    benchmark_tickers = get_benchmark_universe()
    all_tickers = ccteu_tickers + benchmark_tickers
    
    # Initialize Bloomberg API
    logger.info("Initializing Bloomberg connection...")
    api = BloombergAPI()
    
    # Extract data
    price_data = extract_data(api, all_tickers, start_date, end_date)
    
    # Build feature set
    logger.info("Building feature set...")
    features_df = build_feature_set(price_data, ccteu_tickers)
    
    # Save processed data
    features_df.to_csv('data/processed/features.csv')
    logger.info("Features saved to data/processed/features.csv")
    
    # Model training for each CCTeu
    results = {}
    
    for ticker in ccteu_tickers:
        logger.info(f"Training model for {ticker}...")
        
        target_col = f"ret_{ticker}"
        feature_cols = [
            'ret_btp', 'ret_bund', 'ret_euribor', 'ret_spread_btp_bund',
            'z_spread_btp_bund', 'z_euribor'
        ]
        
        # Filter features that exist in dataframe
        available_features = [col for col in feature_cols if col in features_df.columns]
        
        if target_col in features_df.columns and len(available_features) > 0:
            model, predictions, metrics = train_model(
                features_df, 
                target_col, 
                available_features,
                model_type="ridge"
            )
            
            results[ticker] = {
                'model': model,
                'predictions': predictions,
                'metrics': metrics
            }
            
            logger.info(f"{ticker} - R²: {metrics['R2']:.4f}, RMSE: {metrics['RMSE']:.6f}")
    
    # Relative value analysis
    logger.info("Computing relative value signals...")
    
    # Extract CCTeu returns for relative value analysis
    ccteu_returns = features_df[[col for col in features_df.columns if col.startswith('ret_') and any(ticker in col for ticker in ccteu_tickers)]]
    
    if not ccteu_returns.empty:
        rv_signals = compute_relative_value_signals(ccteu_returns)
        rv_signals.to_csv('data/processed/relative_value_signals.csv')
        logger.info("Relative value signals saved to data/processed/relative_value_signals.csv")
    
    # Generate summary report
    generate_summary_report(results, features_df, rv_signals if not ccteu_returns.empty else None)
    
    logger.info("Pricing model execution completed successfully!")


def generate_summary_report(results: dict, features_df: pd.DataFrame, rv_signals: pd.DataFrame = None):
    """Generate summary report of model results."""
    
    report_lines = [
        "=" * 60,
        "CCTeu PRICING MODEL - SUMMARY REPORT",
        "=" * 60,
        f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Data Period: {features_df.index[0].strftime('%Y-%m-%d')} to {features_df.index[-1].strftime('%Y-%m-%d')}",
        f"Total Observations: {len(features_df)}",
        "",
        "MODEL PERFORMANCE METRICS:",
        "-" * 30,
    ]
    
    # Add model metrics
    for ticker, result in results.items():
        metrics = result['metrics']
        report_lines.append(f"{ticker:20} | R²: {metrics['R2']:7.4f} | RMSE: {metrics['RMSE']:8.6f}")
    
    # Add average performance
    if results:
        avg_r2 = np.mean([r['metrics']['R2'] for r in results.values()])
        avg_rmse = np.mean([r['metrics']['RMSE'] for r in results.values()])
        report_lines.extend([
            "-" * 50,
            f"{'AVERAGE':20} | R²: {avg_r2:7.4f} | RMSE: {avg_rmse:8.6f}",
            ""
        ])
    
    # Add relative value summary
    if rv_signals is not None and not rv_signals.empty:
        report_lines.extend([
            "RELATIVE VALUE SIGNALS:",
            "-" * 30,
            f"Most Undervalued: {rv_signals.iloc[-1].idxmin()} (Z-Score: {rv_signals.iloc[-1].min():.2f})",
            f"Most Overvalued: {rv_signals.iloc[-1].idxmax()} (Z-Score: {rv_signals.iloc[-1].max():.2f})",
            ""
        ])
    
    report_lines.append("=" * 60)
    
    # Write report
    with open('data/processed/summary_report.txt', 'w') as f:
        f.write('\n'.join(report_lines))
    
    # Print to console
    print('\n'.join(report_lines))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="CCTeu Pricing Model")
    parser.add_argument('--config', default='config/api_keys.yaml', help='Configuration file path')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set date range
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    else:
        end_date = datetime.now()
    
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    else:
        start_date = end_date - timedelta(days=config['model_config']['default_lookback_days'])
    
    try:
        run_pricing_model(config, start_date, end_date)
    except Exception as e:
        logger.error(f"Model execution failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
