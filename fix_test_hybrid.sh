#!/bin/bash
# Vérifie et corrige les erreurs de syntaxe dans test_hybrid.py

# Backup du fichier original
cp tests/ai/test_hybrid.py tests/ai/test_hybrid.py.bak

# Écrit la version corrigée
cat > tests/ai/test_hybrid.py <<EOL
import pytest
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class TestHybridAI:
    """Tests complets pour le système hybride"""
    
    @pytest.fixture
    def sample_rl_env(self):
        """Environnement de test pour RL"""
        env = gym.make('CartPole-v1')
        yield env
        env.close()
    
    def test_ppo_model(self, sample_rl_env):
        """Teste le modèle PPO"""
        try:
            model = PPO('MlpPolicy', sample_rl_env, verbose=0)
            assert model is not None
            
            model.learn(total_timesteps=100)
        except Exception as e:
            pytest.fail(f"Erreur PPO: {str(e)}")
        
    def test_keras_integration(self):
        """Teste l'intégration Keras"""
        try:
            model = Sequential([
                Dense(64, activation='relu', input_shape=(10,)),
                Dense(1)
            ])
            prediction = model.predict(np.random.rand(1, 10))
            assert prediction.shape == (1, 1)
        except Exception as e:
            pytest.fail(f"Erreur Keras: {str(e)}")

def test_import():
    """Test basique d'importation"""
    try:
        import tensorflow
        import stable_baselines3
        import gymnasium
        assert True
    except ImportError as e:
        pytest.fail(f"Import manquant: {str(e)}")
EOL

echo "Correction terminée. Fichier original sauvegardé dans tests/ai/test_hybrid.py.bak"
