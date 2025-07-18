import tensorflow as tf
import numpy as np
import time

class HybridModel:
    def __init__(self):
        self.model = self._build_debug_model()
        
    def _build_debug_model(self):
        """Modèle de debug avec suivi manuel des métriques"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, input_shape=(30, 5), activation='sigmoid')
        ])
        
        # Custom metrics tracking
        self.train_acc_metric = tf.keras.metrics.BinaryAccuracy()
        self.val_acc_metric = tf.keras.metrics.BinaryAccuracy()
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss='binary_crossentropy'
        )
        return model

    def train_and_validate(self, X_train, y_train, X_val, y_val, epochs=1):
        """Processus d'entraînement manuel"""
        # Reset metrics
        self.train_acc_metric.reset_states()
        self.val_acc_metric.reset_states()
        
        # Entraînement
        self.model.fit(X_train, y_train, epochs=epochs, verbose=0)
        
        # Calcul manuel des métriques
        train_preds = self.model.predict(X_train)
        self.train_acc_metric.update_state(y_train, train_preds)
        
        val_preds = self.model.predict(X_val)
        self.val_acc_metric.update_state(y_val, val_preds)
        
        return {
            'train_acc': float(self.train_acc_metric.result()),
            'val_acc': float(self.val_acc_metric.result())
        }

class HybridAI(HybridModel):
    def __init__(self):
        super().__init__()
        print("🐛 Mode Debug Activé | Métriques Manuelles")