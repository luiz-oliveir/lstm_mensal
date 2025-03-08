import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import pickle
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

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
    def __init__(self, timesteps=7, n_features=1, latent_dim=32):
        super(LSTM_VAE, self).__init__()
        self.timesteps = timesteps
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.encoder_lstm_units = 64
        self.decoder_lstm_units = 64
        
        # Encoder layers
        self.encoder_lstm = tf.keras.layers.LSTM(
            self.encoder_lstm_units,
            return_sequences=True,
            name='encoder_lstm'
        )
        
        self.encoder_mean = tf.keras.layers.Dense(
            self.latent_dim,
            name='encoder_mean'
        )
        
        self.encoder_log_var = tf.keras.layers.Dense(
            self.latent_dim,
            name='encoder_log_var'
        )
        
        # Reparameterization layer
        self.reparameterize_layer = ReparameterizationLayer(name='reparameterize')
        
        # Decoder layers
        self.repeat_vector = RepeatVectorLayer(
            timesteps=self.timesteps,
            name='repeat_vector'
        )
        
        self.decoder_lstm = tf.keras.layers.LSTM(
            self.decoder_lstm_units,
            return_sequences=True,
            name='decoder_lstm'
        )
        
        self.decoder_dense = tf.keras.layers.Dense(
            self.n_features,
            name='decoder_dense'
        )
        
        # Temperature weight layer
        self.temp_weight_layer = TemperatureWeightLayer(name='temp_weight')
        
        # Loss tracking
        self.total_loss_tracker = tf.keras.metrics.Mean(name='total_loss')
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name='reconstruction_loss')
        self.kl_loss_tracker = tf.keras.metrics.Mean(name='kl_loss')
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker
        ]
    
    def encode(self, x):
        encoder_output = self.encoder_lstm(x)
        z_mean = self.encoder_mean(encoder_output[:, -1, :])
        z_log_var = self.encoder_log_var(encoder_output[:, -1, :])
        return z_mean, z_log_var
    
    def decode(self, z):
        x = self.repeat_vector(z)
        x = self.decoder_lstm(x)
        return self.decoder_dense(x)
    
    def call(self, inputs, training=None):
        z_mean, z_log_var = self.encode(inputs)
        z = self.reparameterize_layer([z_mean, z_log_var])
        reconstructed = self.decode(z)
        
        if training:
            # Reconstruction loss with temperature weights
            temp_weights = self.temp_weight_layer(inputs)
            reconstruction_loss = tf.reduce_mean(
                temp_weights * tf.square(inputs - reconstructed)
            )
            
            # KL loss
            kl_loss = -0.5 * tf.reduce_mean(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            )
            
            # Total loss
            total_loss = reconstruction_loss + 0.1 * kl_loss
            
            # Update metrics
            self.total_loss_tracker.update_state(total_loss)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.kl_loss_tracker.update_state(kl_loss)
            
            self.add_loss(total_loss)
        
        return reconstructed
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            reconstructed = self(data, training=True)
        
        # Get trainable variables
        trainable_vars = self.trainable_variables
        
        # Calculate gradients
        gradients = tape.gradient(self.losses, trainable_vars)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        return {
            'loss': self.total_loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result()
        }
    
    def get_config(self):
        config = super(LSTM_VAE, self).get_config()
        config.update({
            'timesteps': self.timesteps,
            'n_features': self.n_features,
            'latent_dim': self.latent_dim,
            'encoder_lstm_units': self.encoder_lstm_units,
            'decoder_lstm_units': self.decoder_lstm_units
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

def prepare_sequences(data, seq_length):
    """Prepare sequences for LSTM VAE with improved preprocessing"""
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)

def train_monthly_model(sequences, timesteps=7, n_features=1, batch_size=64, 
                       latent_dim=32, epochs=100):
    """Train LSTM VAE model for a specific month"""
    try:
        # Create and compile model
        model = LSTM_VAE(
            timesteps=timesteps,
            n_features=n_features,
            latent_dim=latent_dim
        )
        
        # Configure optimizer with gradient clipping
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,
            clipnorm=1.0
        )
        
        # Compile model
        model.compile(optimizer=optimizer)
        
        # Configure callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
        
        # Train model
        history = model.fit(
            sequences,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return model
        
    except Exception as e:
        logging.error(f"Error in train_monthly_model: {str(e)}")
        return None

def train_monthly_models(data_path, output_dir, seq_length=7, batch_size=64, 
                        latent_dim=32, epochs=100):
    """Train separate LSTM VAE models for each month with improved extreme handling"""
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Define month name mapping
        month_map = {
            'jan': 'jan', 'feb': 'fev', 'mar': 'mar', 'abr': 'abr',
            'may': 'mai', 'jun': 'jun', 'jul': 'jul', 'ago': 'ago',
            'sep': 'set', 'oct': 'out', 'nov': 'nov', 'dec': 'dez'
        }
        
        # Load and preprocess data
        df = pd.read_excel(data_path)
        df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y')  # Brazilian date format
        df['month'] = df['Data'].dt.strftime('%b').str.lower()
        df['month'] = df['month'].map(month_map)  # Convert to Portuguese month names
        
        # Process each month
        for month_en, month_pt in month_map.items():
            logging.info(f"Training model for month: {month_pt}")
            
            try:
                # Filter data for current month
                month_data = df[df['month'] == month_pt]['Temperatura Maxima'].values
                
                # Check if we have enough data
                if len(month_data) < seq_length + 1:
                    logging.warning(f"Insufficient data for month {month_pt}. Skipping...")
                    continue
                
                # Reshape data for scaling
                month_data = month_data.reshape(-1, 1)
                
                # Scale data
                scaler = RobustScaler()
                scaled_data = scaler.fit_transform(month_data)
                
                # Prepare sequences
                sequences = prepare_sequences(scaled_data, seq_length)
                
                # Check if we have enough sequences
                if len(sequences) < batch_size:
                    logging.warning(f"Not enough sequences for month {month_pt}. Minimum required: {batch_size}. Got: {len(sequences)}. Skipping...")
                    continue
                
                # Convert to numpy array
                sequences = np.array(sequences)
                
                # Train model
                model = train_monthly_model(
                    sequences=sequences,
                    timesteps=seq_length,
                    n_features=1,
                    batch_size=batch_size,
                    latent_dim=latent_dim,
                    epochs=epochs
                )
                
                if model is not None:
                    # Create model directory if it doesn't exist
                    os.makedirs(os.path.join(output_dir), exist_ok=True)
                    
                    # Save model and scaler
                    model_path = os.path.join(output_dir, f'lstm_vae_model_{month_pt}.keras')
                    scaler_path = os.path.join(output_dir, f'scaler_{month_pt}.pkl')
                    
                    model.save(model_path)
                    with open(scaler_path, 'wb') as f:
                        pickle.dump(scaler, f)
                    
                    logging.info(f"Successfully trained and saved model for {month_pt}")
                else:
                    logging.warning(f"Model training failed for month {month_pt}")
            
            except Exception as e:
                logging.error(f"Error training model for {month_pt}: {str(e)}")
                continue
        
        logging.info("Finished training all monthly models")
        
    except Exception as e:
        logging.error(f"Error training models: {str(e)}")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Set paths
    data_path = "../Convencionais processadas temperaturas/82024.xlsx"
    output_dir = "./lstm_vae_model"
    
    # Train models
    train_monthly_models(
        data_path=data_path,
        output_dir=output_dir,
        seq_length=7,
        batch_size=64,
        latent_dim=32,
        epochs=100
    )
