#!/bin/bash
# Correction complète pour test_hybrid.py

echo "1. Création des fichiers de correctifs..."
cat <<EOL > tf_spec_fix.py
import tensorflow as tf
if not hasattr(tf, '__spec__'):
    tf.__spec__ = "tensorflow"
EOL

cat <<EOL > keras_compat.py
from tensorflow.keras import backend as K
if not hasattr(K, 'batch_outputs'):
    K.batch_outputs = []
EOL

echo "2. Application des correctifs..."
cp tests/ai/test_hybrid.py tests/ai/test_hybrid.py.bak
cat <<EOL > tests/ai/test_hybrid.py
import tf_spec_fix
import keras_compat
import tf_spec_fix
import keras_compat
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
            # Workaround pour TensorFlow spec
            import tensorflow as tf
            if not hasattr(tf, '__spec__'):
                tf.__spec__ = "tensorflow"
                
            model = PPO('MlpPolicy', sample_rl_env, verbose=0)
            assert model is not None
            
            # Test d'apprentissage basique
            model.learn(total_timesteps=100)
        except Exception as e:
            pytest.fail(f"Erreur PPO: {str(e)}")
        
    def test_keras_integration(self):
        """Teste l'intégration Keras"""
        try:
            # Workaround pour batch_outputs
            from tensorflow.keras import backend as K
            if not hasattr(K, 'batch_outputs'):
                K.batch_outputs = []
                
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

echo "3. Vérification des installations..."
pip install --upgrade tensorflow stable-baselines3 gymnasium

echo "Correction terminée. Backup sauvegardé dans tests/ai/test_hybrid.py.bak"
echo "Exécutez les tests avec: python -m pytest tests/ai/test_hybrid.py -v"
