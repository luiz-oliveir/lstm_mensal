import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import glob
import datetime
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lstm_vae_mensal.log'),
        logging.StreamHandler()
    ]
)

# Configurações
row_mark = 740
batch_size = 128
timesteps = 7  # Janela de tempo para análise
n_features = 1  # Número de features (temperatura)
latent_dim = 32  # Dimensão do espaço latente
epoch_num = 100
threshold = None

# Diretórios
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.dirname(base_dir), "Convencionais processadas temperaturas")
model_dir = os.path.join(base_dir, "lstm_vae_model")
images_dir = os.path.join(base_dir, "lstm_vae_images")
results_dir = os.path.join(base_dir, "Resumo resultados")

# Verificar e criar diretórios
for dir_path in [model_dir, images_dir, results_dir]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logging.info(f"Directory created: {dir_path}")
    else:
        logging.info(f"Directory verified: {dir_path}")

# Dicionário de meses
meses = {
    1:'jan', 2:'fev', 3:'mar', 4:'abr', 5:'mai', 6:'jun',
    7:'jul', 8:'ago', 9:'set', 10:'out', 11:'nov', 12:'dez'
}

class ReparameterizationLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ReparameterizationLayer, self).__init__(**kwargs)
    
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch_size = tf.shape(z_mean)[0]
        latent_dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch_size, latent_dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class RepeatVectorLayer(tf.keras.layers.Layer):
    def __init__(self, timesteps, **kwargs):
        super(RepeatVectorLayer, self).__init__(**kwargs)
        self.timesteps = timesteps
    
    def call(self, inputs):
        return tf.repeat(tf.expand_dims(inputs, axis=1), repeats=self.timesteps, axis=1)
    
    def get_config(self):
        config = super(RepeatVectorLayer, self).get_config()
        config.update({'timesteps': self.timesteps})
        return config

class TemperatureWeightLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(TemperatureWeightLayer, self).__init__(**kwargs)
    
    def call(self, x):
        mean = tf.keras.backend.mean(x, axis=[1, 2], keepdims=True)
        std = tf.keras.backend.std(x, axis=[1, 2], keepdims=True) + tf.keras.backend.epsilon()
        z_scores = tf.abs((x - mean) / std)
        weights = tf.exp(z_scores)
        weights = weights / (tf.keras.backend.mean(weights, axis=[1, 2], keepdims=True) + tf.keras.backend.epsilon())
        return weights

class LSTM_VAE(tf.keras.Model):
    def __init__(self, timesteps=7, n_features=1, latent_dim=32, **kwargs):
        super(LSTM_VAE, self).__init__(**kwargs)
        self.timesteps = timesteps
        self.n_features = n_features
        self.latent_dim = latent_dim
        
        # First LSTM layer
        self.lstm = tf.keras.layers.LSTM(32, return_sequences=True, name='lstm')  # Output shape: (batch_size, timesteps, 32)
        
        # First LSTM output processing
        self.dense = tf.keras.layers.Dense(4, name='dense')  # Output shape: (batch_size, timesteps, 4)
        self.dense_1 = tf.keras.layers.Dense(4, name='dense_1')  # Output shape: (batch_size, timesteps, 4)
        self.dense_2 = tf.keras.layers.Dense(32, name='dense_2')  # Output shape: (batch_size, timesteps, 32)
        
        # Second LSTM layer
        self.lstm_1 = tf.keras.layers.LSTM(32, return_sequences=True, name='lstm_1')  # Output shape: (batch_size, timesteps, 32)
        
        # Second LSTM output processing
        self.dense_3 = tf.keras.layers.Dense(32, name='dense_3')  # Output shape: (batch_size, timesteps, 32)
        self.dense_4 = tf.keras.layers.Dense(16, name='dense_4')  # Output shape: (batch_size, timesteps, 16)
        self.dense_5 = tf.keras.layers.Dense(1, name='dense_5')  # Output shape: (batch_size, timesteps, 1)
        
        # Final processing branch
        self.dense_6 = tf.keras.layers.Dense(16, name='dense_6')  # Output shape: (batch_size, timesteps, 16)
        self.dense_7 = tf.keras.layers.Dense(1, name='dense_7')  # Output shape: (batch_size, timesteps, 1)
        
        # Additional layers
        self.repeat_vector = tf.keras.layers.RepeatVector(timesteps, name='repeat_vector')  # Output shape: (batch_size, timesteps, 32)
        self.dropout = tf.keras.layers.Dropout(0.2, name='dropout')  # Output shape: (batch_size, timesteps, 16)
        self.layer_norm = tf.keras.layers.LayerNormalization(name='layer_normalization')  # Output shape: (batch_size, timesteps, 16)
    
    def call(self, inputs, training=None):
        # First LSTM processing
        x = self.lstm(inputs)  # Output shape: (batch_size, timesteps, 32)
        
        # Process first LSTM output
        x = self.dense(x)  # Output shape: (batch_size, timesteps, 4)
        x = self.dense_1(x)  # Output shape: (batch_size, timesteps, 4)
        x = self.dense_2(x)  # Output shape: (batch_size, timesteps, 32)
        
        # Second LSTM processing
        x = self.lstm_1(x)  # Output shape: (batch_size, timesteps, 32)
        
        # Get last timestep for repeat vector
        last_timestep = x[:, -1, :]  # Output shape: (batch_size, 32)
        x = self.repeat_vector(last_timestep)  # Output shape: (batch_size, timesteps, 32)
        
        # Process repeated vector
        x = self.dense_3(x)  # Output shape: (batch_size, timesteps, 32)
        x = self.dense_4(x)  # Output shape: (batch_size, timesteps, 16)
        
        # Apply dropout during training
        if training:
            x = self.dropout(x)  # Output shape: (batch_size, timesteps, 16)
        
        # Apply layer normalization
        x = self.layer_norm(x)  # Output shape: (batch_size, timesteps, 16)
        
        # Generate outputs through two branches
        output1 = self.dense_5(x)  # Output shape: (batch_size, timesteps, 1)
        x = self.dense_6(x)  # Output shape: (batch_size, timesteps, 16)
        output2 = self.dense_7(x)  # Output shape: (batch_size, timesteps, 1)
        
        # Combine outputs
        outputs = tf.concat([output1, output2], axis=-1)  # Output shape: (batch_size, timesteps, 2)
        
        return outputs
    
    def get_config(self):
        config = super(LSTM_VAE, self).get_config()
        config.update({
            'timesteps': self.timesteps,
            'n_features': self.n_features,
            'latent_dim': self.latent_dim
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

def reshape(da):
    """Reshape dados para formato LSTM"""
    data = []
    for i in range(len(da) - timesteps + 1):
        data.append(da[i:(i + timesteps)])
    return np.array(data)

def prepare_training_data(data, batch_size=128):
    """Prepara dados para o modelo"""
    data = reshape(data)
    data = data.reshape(-1, timesteps, n_features)
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
    return dataset

def calculate_advanced_metrics(predictions, originals):
    """Calculate advanced evaluation metrics"""
    mse = np.mean(np.square(predictions - originals))
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - originals))
    
    # Evitar divisão por zero
    mape = np.mean(np.abs((originals - predictions) / (originals + 1e-6))) * 100
    
    # Métricas específicas para valores extremos
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    orig_percentiles = np.percentile(originals, percentiles)
    pred_percentiles = np.percentile(predictions, percentiles)
    
    percentile_errors = {
        f'p{p}_error': abs(o - p)
        for p, o, p in zip(percentiles, orig_percentiles, pred_percentiles)
    }
    
    # Log likelihood aproximado (assumindo distribuição normal)
    residuals = predictions - originals
    std = np.std(residuals) + 1e-6
    log_px = -0.5 * np.mean(np.square(residuals) / std + np.log(2 * np.pi * std))
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'log_px': log_px,
        **percentile_errors
    }
    
    return metrics, log_px

def save_monthly_results(predictions, originals, log_px, df_original, file_name, mes):
    """Save monthly analysis results to Excel file with multiple sheets"""
    try:
        # Create output filename
        base_name = os.path.splitext(file_name)[0]
        output_file = os.path.join(results_dir, f'analise_mensal_{mes}_{base_name}.xlsx')
        
        # Calculate metrics
        metrics, _ = calculate_advanced_metrics(predictions, originals)
        
        # Calculate monthly statistics
        monthly_stats = calcular_estatisticas_mensais(df_original, predictions, originals)
        
        # Create DataFrames for each sheet
        summary_df = pd.DataFrame({
            'Metric': metrics.keys(),
            'Value': metrics.values()
        })
        
        detailed_df = pd.DataFrame({
            'Data': df_original.index,
            'Original': originals,
            'Predicted': predictions,
            'Log_Likelihood': log_px,
            'Error': originals - predictions,
            'Error_Abs': np.abs(originals - predictions),
            'Error_Pct': np.abs((originals - predictions) / (originals + 1e-6)) * 100,
            'Year': df_original.index.year,
            'Month': df_original.index.month
        })
        
        # Análise anual
        annual_stats = detailed_df.groupby('Year').agg({
            'Original': ['mean', 'std', 'min', 'max', lambda x: np.percentile(x, 5), lambda x: np.percentile(x, 95)],
            'Predicted': ['mean', 'std', 'min', 'max', lambda x: np.percentile(x, 5), lambda x: np.percentile(x, 95)],
            'Error': ['mean', 'std'],
            'Error_Abs': ['mean', 'max'],
            'Error_Pct': ['mean', 'max'],
            'Log_Likelihood': ['mean', 'std', 'min']
        })
        
        # Rename columns for better readability
        annual_stats.columns = [
            'Original_Mean', 'Original_Std', 'Original_Min', 'Original_Max', 'Original_P5', 'Original_P95',
            'Predicted_Mean', 'Predicted_Std', 'Predicted_Min', 'Predicted_Max', 'Predicted_P5', 'Predicted_P95',
            'Error_Mean', 'Error_Std',
            'AbsError_Mean', 'AbsError_Max',
            'PctError_Mean', 'PctError_Max',
            'LogLik_Mean', 'LogLik_Std', 'LogLik_Min'
        ]
        
        # Save all DataFrames to Excel
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Resumo', index=False)
            detailed_df.to_excel(writer, sheet_name='Dados Detalhados', index=True)
            pd.DataFrame(monthly_stats).transpose().to_excel(writer, sheet_name='Estatísticas Mensais')
            annual_stats.to_excel(writer, sheet_name='Análise Anual')
        
        logging.info(f"Results saved to {output_file}")
        
    except Exception as e:
        logging.error(f"Error saving results to Excel: {str(e)}")
        raise

def load_model_and_scaler(month):
    """Load model and scaler for a specific month"""
    try:
        # Create model with custom object scope
        with tf.keras.utils.custom_object_scope({
            'LSTM_VAE': LSTM_VAE,
            'ReparameterizationLayer': ReparameterizationLayer,
            'RepeatVectorLayer': RepeatVectorLayer,
            'TemperatureWeightLayer': TemperatureWeightLayer
        }):
            # Create new LSTM_VAE instance
            model = LSTM_VAE(timesteps=timesteps, n_features=n_features, latent_dim=latent_dim)
            
            # Build model with input shape
            dummy_input = tf.zeros((1, timesteps, n_features))
            _ = model(dummy_input, training=False)
            
            # Load weights
            model_path = os.path.join(model_dir, f'lstm_vae_model_{month}.h5')
            model.load_weights(model_path)
            
            # Load scaler
            scaler_path = os.path.join(model_dir, f'scaler_{month}.pkl')
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            
            logging.info(f"Successfully loaded model and scaler for month {month}")
            
            return model, scaler
            
    except Exception as e:
        logging.error(f"Error loading model/scaler for month {month}: {str(e)}")
        raise

def calcular_estatisticas_mensais(df_original, predictions, originals):
    """Calculate detailed monthly statistics"""
    try:
        # Estatísticas básicas
        stats = {
            'Média Original': np.mean(originals),
            'Desvio Padrão Original': np.std(originals),
            'Mínimo Original': np.min(originals),
            'Máximo Original': np.max(originals),
            'Média Prevista': np.mean(predictions),
            'Desvio Padrão Previsto': np.std(predictions),
            'Mínimo Previsto': np.min(predictions),
            'Máximo Previsto': np.max(predictions)
        }
        
        # Análise de valores extremos
        percentiles = [1, 5, 25, 50, 75, 95, 99]
        for p in percentiles:
            stats[f'Percentil {p}% Original'] = np.percentile(originals, p)
            stats[f'Percentil {p}% Previsto'] = np.percentile(predictions, p)
        
        # Análise de erro por faixa de temperatura
        mean_temp = np.mean(originals)
        std_temp = np.std(originals)
        
        # Definir faixas de temperatura
        ranges = [
            ('Extremamente Baixo', lambda x: x < mean_temp - 2*std_temp),
            ('Muito Baixo', lambda x: (x >= mean_temp - 2*std_temp) & (x < mean_temp - std_temp)),
            ('Baixo', lambda x: (x >= mean_temp - std_temp) & (x < mean_temp - 0.5*std_temp)),
            ('Normal', lambda x: (x >= mean_temp - 0.5*std_temp) & (x <= mean_temp + 0.5*std_temp)),
            ('Alto', lambda x: (x > mean_temp + 0.5*std_temp) & (x <= mean_temp + std_temp)),
            ('Muito Alto', lambda x: (x > mean_temp + std_temp) & (x <= mean_temp + 2*std_temp)),
            ('Extremamente Alto', lambda x: x > mean_temp + 2*std_temp)
        ]
        
        # Calcular métricas por faixa
        for range_name, range_func in ranges:
            mask = range_func(originals)
            if np.any(mask):
                range_orig = originals[mask]
                range_pred = predictions[mask]
                
                mse = np.mean(np.square(range_pred - range_orig))
                mae = np.mean(np.abs(range_pred - range_orig))
                mape = np.mean(np.abs((range_orig - range_pred) / (range_orig + 1e-6))) * 100
                
                stats[f'{range_name} - Contagem'] = np.sum(mask)
                stats[f'{range_name} - MSE'] = mse
                stats[f'{range_name} - MAE'] = mae
                stats[f'{range_name} - MAPE'] = mape
        
        return pd.Series(stats)
        
    except Exception as e:
        logging.error(f"Error calculating monthly statistics: {str(e)}")
        raise

def process_file(file_path):
    """Process a single file with monthly analysis"""
    try:
        logging.info(f"Processing file: {file_path}")
        file_name = os.path.basename(file_path)
        
        # Read data
        df = pd.read_excel(file_path)
        
        # Convert date column if needed
        if 'Data' not in df.columns and 'DATA' in df.columns:
            df = df.rename(columns={'DATA': 'Data'})
        
        # Convert date using Brazilian format (dd/mm/yyyy)
        df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y')
        df.set_index('Data', inplace=True)
        
        # Sort index to ensure chronological order
        df = df.sort_index()
        
        # Process each month
        for month in range(1, 13):
            mes = meses[month]
            logging.info(f"Processing month: {mes}")
            
            try:
                # Load model and scaler for the month
                model, scaler = load_model_and_scaler(mes)
                
                # Filter data for the month
                df_month = df[df.index.month == month].copy()
                if len(df_month) < timesteps:
                    logging.warning(f"Insufficient data for month {mes}. Skipping...")
                    continue
                
                # Prepare data
                if 'Temperatura Maxima' in df_month.columns:
                    data = df_month['Temperatura Maxima'].values.reshape(-1, 1)
                elif 'TEMPERATURA MAXIMA' in df_month.columns:
                    data = df_month['TEMPERATURA MAXIMA'].values.reshape(-1, 1)
                else:
                    raise ValueError("Temperature column not found in data")
                
                # Remove any missing values
                if np.any(np.isnan(data)):
                    logging.warning(f"Found {np.sum(np.isnan(data))} missing values in month {mes}")
                    data = data[~np.isnan(data)]
                
                data_scaled = scaler.transform(data)
                
                # Create sequences
                sequences = reshape(data_scaled)
                
                # Generate predictions
                predictions_scaled = model.predict(sequences)
                predictions_scaled = predictions_scaled[:, -1, :]  # Get last timestep
                
                # Inverse transform predictions
                predictions = scaler.inverse_transform(predictions_scaled)
                
                # Calculate log likelihood
                residuals = predictions - data[-len(predictions):]
                std = np.std(residuals) + 1e-6
                log_px = -0.5 * np.square(residuals) / std - 0.5 * np.log(2 * np.pi * std)
                
                # Save results
                save_monthly_results(
                    predictions.flatten(),
                    data[-len(predictions):].flatten(),
                    log_px.flatten(),
                    df_month[-len(predictions):],
                    file_name,
                    mes
                )
                
                logging.info(f"Successfully processed month {mes}")
                
            except Exception as e:
                logging.error(f"Error processing month {mes}: {str(e)}")
                continue
        
        logging.info(f"Finished processing file: {file_path}")
        
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {str(e)}")
        raise

def main():
    """Main function with error handling and progress tracking"""
    try:
        # Get list of Excel files
        excel_files = glob.glob(os.path.join(data_dir, "*.xlsx"))
        
        if not excel_files:
            logging.warning(f"No Excel files found in directory: {data_dir}")
            return
        
        logging.info(f"Found {len(excel_files)} Excel files to process")
        
        # Process each file
        for file_path in tqdm(excel_files, desc="Processing files"):
            try:
                process_file(file_path)
            except Exception as e:
                logging.error(f"Failed to process file {file_path}: {str(e)}")
                continue
        
        logging.info("Finished processing all files")
        
    except Exception as e:
        logging.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    main()
