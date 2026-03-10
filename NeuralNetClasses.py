import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class AutoencoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, encoding_dim=32, epochs=20, batch_size=32, dropout=False):
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout

    def fit(self, X, y=None):
        input_dim = X.shape[1]
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(128, activation='relu')(input_layer)

        if self.dropout:
            encoded = Dropout(0.2)(encoded)
            encoded = Dense(64, activation='relu')(encoded)
            encoded = Dropout(0.2)(encoded)
            bottleneck = Dense(self.encoding_dim, activation='linear')(encoded)
            decoded = Dense(64, activation='relu')(bottleneck)
            decoded = Dropout(0.2)(decoded)
            decoded = Dense(128, activation='relu')(decoded)
            decoded = Dropout(0.2)(decoded)

        else:
            encoded = Dense(64, activation='relu')(encoded)
            #encoded = Dense(32, activation='relu')(encoded)
            bottleneck = Dense(self.encoding_dim, activation='linear')(encoded)
            #encoded = Dense(32, activation='relu')(encoded)
            decoded = Dense(64, activation='relu')(bottleneck)
            decoded = Dense(128, activation='relu')(decoded)
            
        output_layer = Dense(input_dim, activation='linear')(decoded)

        self.autoencoder = Model(inputs=input_layer, outputs=output_layer)
        self.encoder = Model(inputs=input_layer, outputs=bottleneck)

        self.autoencoder.compile(optimizer=Adam(), loss='mse')
        self.autoencoder.fit(X, X, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return self

    def transform(self, X):
        return self.encoder.predict(X)

# new for proteomics

class AutoencoderTransformerProt(BaseEstimator, TransformerMixin):
    def __init__(self, encoding_dim=32, epochs=20, batch_size=32, dropout=False, early_stop=True, patience=10):
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.early_stop = early_stop
        self.patience = patience

    def fit(self, X, y=None):
        input_dim = X.shape[1]
        input_layer = Input(shape=(input_dim,))

        first_layer_size = 128 if input_dim <=512 else 512 if input_dim <= 1024 else 1024

        encoded = Dense(first_layer_size, activation='relu')(input_layer)

        if self.dropout:
            
            encoded = Dropout(0.2)(encoded)

            if first_layer_size == 1024:
                encoded = Dense(512, activation='relu')(encoded)
                encoded = Dropout(0.2)(encoded)
            
            if first_layer_size >= 512:
                encoded = Dense(256, activation='relu')(encoded)
                encoded = Dropout(0.2)(encoded)
                encoded = Dense(128, activation='relu')(encoded)
                encoded = Dropout(0.2)(encoded)
            
            encoded = Dense(64, activation='relu')(encoded)
            encoded = Dropout(0.2)(encoded)
            
            bottleneck = Dense(self.encoding_dim, activation='linear')(encoded)
            
            decoded = Dense(64, activation='relu')(bottleneck)
            decoded = Dropout(0.2)(decoded)
            decoded = Dense(128, activation='relu')(decoded)
            decoded = Dropout(0.2)(decoded)
        
            if first_layer_size >=512:
                
                decoded = Dense(256, activation='relu')(decoded)
                decoded = Dropout(0.2)(decoded)
                decoded = Dense(512, activation='relu')(decoded)
                decoded = Dropout(0.2)(decoded)

            if first_layer_size == 1024:
                decoded = Dense(1024, activation='relu')(decoded)
                decoded = Dropout(0.2)(decoded)

        else:
            if first_layer_size == 1024:
                encoded = Dense(512, activation='relu')(encoded)
            if first_layer_size >=512:
                encoded = Dense(256, activation='relu')(encoded)
                encoded = Dense(128, activation='relu')(encoded)
            
            encoded = Dense(64, activation='relu')(encoded)

            bottleneck = Dense(self.encoding_dim, activation='linear')(encoded)
            
            decoded = Dense(64, activation='relu')(bottleneck)

            if first_layer_size >=512:
                decoded = Dense(128, activation='relu')(decoded)
                decoded = Dense(256, activation='relu')(decoded)
                decoded = Dense(512, activation='relu')(decoded)
            if first_layer_size == 1024:
                decoded = Dense(1024, activation='relu')(decoded)
            
        output_layer = Dense(input_dim, activation='linear')(decoded)

        self.autoencoder = Model(inputs=input_layer, outputs=output_layer)
        self.encoder = Model(inputs=input_layer, outputs=bottleneck)

        self.autoencoder.compile(optimizer=Adam(), loss='mse')

        callbacks = []
        if self.early_stop:
            early_stop = EarlyStopping(
                monitor='val_loss', 
                patience=self.patience, 
                restore_best_weights=True)
            callbacks.append(early_stop)

        history = self.autoencoder.fit(
            X, X, 
            epochs=self.epochs, 
            batch_size=self.batch_size, 
            validation_split=0.1,
            callbacks=callbacks,
            verbose=0)
        
        self.n_epochs_trained = len(history.history['loss'])
        return self

    def transform(self, X):
        return self.encoder.predict(X)