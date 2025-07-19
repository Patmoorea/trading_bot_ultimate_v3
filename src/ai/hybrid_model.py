import tensorflow as tf
import numpy as np

# Configuration nucléaire pour la reproductibilité
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

class HybridModel:
    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(30, 5)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation='sigmoid', 
                                kernel_initializer='zeros')
        ])
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Données triviales 100% séparables
        self.X = np.array([
            np.ones((30, 5)),    # Classe 1
            -np.ones((30, 5))    # Classe 0
        ]).astype(np.float32)
        self.y = np.array([1, 0], dtype=np.float32)

    def validate(self):
        _, acc = self.model.evaluate(self.X, self.y, verbose=0)
        print(f"Accuracy: {acc} (doit être 1.0)")
        return acc

class HybridAI(HybridModel):
    def __init__(self):
        super().__init__()
        print("🔐 Environnement 100% déterministe initialisé")