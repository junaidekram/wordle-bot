import os
import json
from appwrite.client import Client
from appwrite.services.storage import Storage
from appwrite.services.databases import Databases
from appwrite.id import ID
import tempfile
from wordle_env import WordleEnv
from agent import create_model, WordleGymEnv, TrainingCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# Appwrite configuration from environment variables
client = Client()
client.set_endpoint(os.environ['APPWRITE_FUNCTION_API_ENDPOINT'])
client.set_project(os.environ['APPWRITE_FUNCTION_PROJECT_ID'])
client.set_key(os.environ['APPWRITE_API_KEY'])

storage = Storage(client)
databases = Databases(client)

DATABASE_ID = 'wordle_bot_db'
COLLECTION_TRAINING_LOGS = 'training_logs'
BUCKET_MODELS = 'models'

def load_word_lists():
    # In Appwrite Function, you can bundle word list files or fetch from Storage/Database
    with open('word_lists/target_words_5.txt', 'r') as f:
        target_words = [line.strip() for line in f]
    with open('word_lists/guessable_words_5.txt', 'r') as f:
        guessable_words = [line.strip() for line in f]
    return target_words, guessable_words

def main(context):
    # Parse trigger (could be scheduled or manual)
    word_length = int(context.req.query.get('length', 5))
    n_games = int(context.req.query.get('games', 1000))
    
    target_words, guessable_words = load_word_lists()
    
    env = DummyVecEnv([lambda: WordleGymEnv(word_length, target_words, guessable_words)])
    
    # Try to load existing model from Appwrite Storage
    model_file_id = 'latest_model'  # You'd maintain this in Database
    model_path = None
    try:
        # Download model from storage
        result = storage.get_file_download(BUCKET_MODELS, model_file_id)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp:
            tmp.write(result)
            model_path = tmp.name
    except:
        pass
    
    model = create_model(env, model_path)
    
    # Train
    callback = TrainingCallback()
    model.learn(total_timesteps=n_games * 6, callback=callback)  # Roughly n_games
    
    # Save model locally
    temp_model_path = tempfile.mktemp(suffix='.zip')
    model.save(temp_model_path)
    
    # Upload to Appwrite Storage
    with open(temp_model_path, 'rb') as f:
        result = storage.create_file(
            bucket_id=BUCKET_MODELS,
            file_id=ID.unique(),
            file=f
        )
    
    # Update database with new model ID and training stats
    databases.create_document(
        database_id=DATABASE_ID,
        collection_id=COLLECTION_TRAINING_LOGS,
        document_id=ID.unique(),
        data={
            'timestamp': context.req.headers.get('x-appwrite-timestamp'),
            'word_length': word_length,
            'games_played': n_games,
            'win_rate': callback.win_rates[-1] if callback.win_rates else 0,
            'model_file_id': result['$id']
        }
    )
    
    return context.res.json({'status': 'training completed', 'model_id': result['$id']})