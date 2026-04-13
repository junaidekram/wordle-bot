import os
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
from gym import spaces
import numpy as np
from wordle_env import WordleEnv

class WordleGymEnv(gym.Env):
    """Wrap WordleEnv to be gym-compatible."""
    def __init__(self, word_length=5, target_words=None, guessable_words=None):
        super().__init__()
        self.env = WordleEnv(word_length, target_words=target_words, guessable_words=guessable_words)
        self.action_space = spaces.Discrete(len(guessable_words))
        self.observation_space = spaces.Box(low=-1, high=2, shape=(26 * word_length + 1,), dtype=np.float32)
        self.guessable_words = guessable_words

    def reset(self):
        return self.env.reset()

    def step(self, action):
        guess = self.guessable_words[action]
        obs, reward, done, info = self.env.step(guess)
        return obs, reward, done, info

class TrainingCallback(BaseCallback):
    """Custom callback to log progress to Appwrite."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.win_rates = []
    
    def _on_step(self):
        if 'won' in self.locals.get('infos', [{}])[0]:
            # Log to Appwrite Database here
            pass
        return True

def create_model(env, model_path=None):
    if model_path and os.path.exists(model_path):
        return DQN.load(model_path, env=env)
    return DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        verbose=1
    )