import random
import numpy as np
from typing import List, Tuple, Dict

class WordleEnv:
    def __init__(self, word_length: int = 5, max_attempts: int = 6,
                 target_words: List[str] = None, guessable_words: List[str] = None):
        self.word_length = word_length
        self.max_attempts = max_attempts
        self.target_words = target_words or []
        self.guessable_words = guessable_words or []
        self.reset()

    def reset(self) -> np.ndarray:
        self.target = random.choice(self.target_words)
        self.attempts = []
        self.guess_count = 0
        self.done = False
        self.won = False
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Convert game state to a numeric vector for the neural network."""
        # State representation: 
        # - Attempts left (1)
        # - Letter feedback matrix (26 letters × word_length positions)
        #   Each cell: 2=green, 1=yellow, 0=unknown, -1=gray
        state = np.zeros(26 * self.word_length + 1, dtype=np.float32)
        state[0] = (self.max_attempts - self.guess_count) / self.max_attempts
        
        # Fill feedback from previous guesses
        for attempt in self.attempts:
            guess, feedback = attempt
            for i, (letter, fb) in enumerate(zip(guess, feedback)):
                letter_idx = ord(letter) - ord('a')
                if fb == 'correct':
                    state[1 + letter_idx * self.word_length + i] = 2.0
                elif fb == 'present':
                    state[1 + letter_idx * self.word_length + i] = 1.0
                elif fb == 'absent':
                    state[1 + letter_idx * self.word_length + i] = -1.0
        return state

    def step(self, guess: str) -> Tuple[np.ndarray, float, bool, Dict]:
        if guess not in self.guessable_words:
            # Invalid guess - penalize heavily
            return self._get_state(), -10.0, False, {'invalid': True}
        
        feedback = self._evaluate_guess(guess, self.target)
        self.attempts.append((guess, feedback))
        self.guess_count += 1
        
        # Calculate reward
        reward = self._calculate_reward(guess, feedback)
        
        if guess == self.target:
            self.done = True
            self.won = True
            reward += 10.0  # Win bonus
        elif self.guess_count >= self.max_attempts:
            self.done = True
            self.won = False
            reward -= 5.0  # Loss penalty
        
        return self._get_state(), reward, self.done, {'won': self.won}

    def _evaluate_guess(self, guess: str, target: str) -> List[str]:
        # Same logic as your JS version
        result = ['absent'] * len(guess)
        target_letters = list(target)
        guess_letters = list(guess)
        
        # First pass: greens
        for i in range(len(guess)):
            if guess[i] == target[i]:
                result[i] = 'correct'
                target_letters[i] = None
                guess_letters[i] = None
        
        # Second pass: yellows
        for i in range(len(guess)):
            if guess_letters[i] is not None:
                try:
                    idx = target_letters.index(guess_letters[i])
                    result[i] = 'present'
                    target_letters[idx] = None
                except ValueError:
                    pass
        return result

    def _calculate_reward(self, guess: str, feedback: List[str]) -> float:
        reward = -0.5  # Small penalty for each guess to encourage efficiency
        # Bonus for new information
        green_count = feedback.count('correct')
        yellow_count = feedback.count('present')
        reward += green_count * 1.0 + yellow_count * 0.5
        return reward