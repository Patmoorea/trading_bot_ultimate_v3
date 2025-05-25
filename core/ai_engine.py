
class EnhancedHybridAI:
    def __init__(self):
        # Nouvelle architecture pour M4
        self.technical_model = CNN_LSTM_M4()  # Optimisé Metal
        self.decision_model = PPO_GTrXL_M4()
        
    def predict(self, data):
        # Nouvelle pipeline M4-optimisée
        with tf.device('/GPU:0'):
            tech_pred = self.technical_model(data)
            alloc = self.decision_model(tech_pred)
        return alloc * 0.95  # Smoothing
