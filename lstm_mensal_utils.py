import os
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from datetime import datetime

def setup_logging():
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def verify_directories(*directories):
    """Verify and create directories if they don't exist"""
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logging.info(f"Directory verified/created: {directory}")
        else:
            logging.info(f"Directory verified/created: {directory}")

def load_data(file_path):
    """Load and preprocess data from Excel file"""
    try:
        df = pd.read_excel(file_path)
        df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y')
        return df
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {str(e)}")
        raise

def split_by_month(df):
    """Split dataframe by month"""
    try:
        months = {
            1: 'jan', 2: 'fev', 3: 'mar', 4: 'abr', 5: 'mai', 6: 'jun',
            7: 'jul', 8: 'ago', 9: 'set', 10: 'out', 11: 'nov', 12: 'dez'
        }
        return {months[month]: group for month, group in df.groupby(df['Data'].dt.month)}
    except Exception as e:
        logging.error(f"Error splitting data by month: {str(e)}")
        raise

def reshape(data, time_step=1):
    """Reshape data for LSTM input"""
    try:
        return np.reshape(data, (data.shape[0], time_step, data.shape[1]))
    except Exception as e:
        logging.error(f"Error reshaping data: {str(e)}")
        raise

def calculate_monthly_statistics(df_mes, predictions, log_likelihood):
    """Calculate monthly statistics"""
    try:
        stats = {
            'Total Records': len(df_mes),
            'Mean Log Likelihood': np.mean(log_likelihood),
            'Std Log Likelihood': np.std(log_likelihood),
            'Min Log Likelihood': np.min(log_likelihood),
            'Max Log Likelihood': np.max(log_likelihood)
        }
        
        # Calculate percentiles
        percentiles = [1, 5, 25, 50, 75, 95, 99]
        for p in percentiles:
            stats[f'{p}th Percentile'] = np.percentile(log_likelihood, p)
            
        # Calculate yearly statistics
        yearly_stats = df_mes.groupby(df_mes['Data'].dt.year).agg({
            'Temperatura Maxima': ['mean', 'std', 'min', 'max']
        })
        
        return stats, yearly_stats
    except Exception as e:
        logging.error(f"Error calculating monthly statistics: {str(e)}")
        raise
