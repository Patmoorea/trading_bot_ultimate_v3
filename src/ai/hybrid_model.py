print("=== FICHIER src/ai/hybrid_model.py CHARGE ===")

import tensorflow as tf
import numpy as np

class HybridAI:
    def __init__(self):
        print("=== HybridAI __init__ appelée ===")
        self.learning_rate = 1e-3  # Correct: dans __init__
        self.cnn_lstm = self.build_cnn_lstm()  # Correct

    def build_cnn_lstm(self):  # Correct: self inclus
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(20, 5)),
            tf.keras.layers.LSTM(8),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def validate(self):  # Correct: self inclus
        print("=== validate() APPELEE ===")
        X = np.random.randn(8, 20, 5)
        y = np.random.randint(0, 2, 8)
        history = self.cnn_lstm.fit(X, y, epochs=1, verbose=0)
        return float(history.history["accuracy"][0])

# Test séparé
if __name__ == "__main__":
    model = HybridAI()
    print("Accuracy:", model.validate())