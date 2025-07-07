print("=== FICHIER src/ai/hybrid_model.py CHARGE ===")

import tensorflow as tf
import numpy as np


class HybridAI:
    def __init__(self):
        print("=== HybridAI __init__ appelée ===")
        self.learning_rate = 1e-3
        self.cnn_lstm = self.build_cnn_lstm()

    def build_cnn_lstm(self):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(20, 5)),
                tf.keras.layers.LSTM(8),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def validate(self):
        print("=== validate() de HybridAI APPELEE avec LR:", self.learning_rate)
        X = np.random.randn(8, 20, 5)
        y = np.random.randint(0, 2, size=(8, 1))
        # Nouvelle façon de setter le learning rate :
        self.cnn_lstm.optimizer.learning_rate.assign(self.learning_rate)
        history = self.cnn_lstm.fit(X, y, epochs=1, batch_size=4, verbose=0)
        acc = float(history.history["accuracy"][-1])
        print("=== ACCURACY =", acc)
        return acc
