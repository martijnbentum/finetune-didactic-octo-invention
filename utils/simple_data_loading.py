from datasets import load_dataset
import json
import librosa
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor

def load_json(filename = '../JSON/example.json'):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def load_dataset_from_json(filename = '../JSON/example.json'):
    dataset = load_dataset('json', data_files=filename, field='data',
        cache_dir = '../example_cache_dir')
    return dataset

def load_audio_on_dataset(dataset = None):
    if dataset is None:
        dataset = load_dataset_from_json()
    dataset = dataset.map(_load_audio)
    return dataset

def load_audio(filename, start = 0.0, end=None):
	if not end: duration = None
	else: duration = end - start
	audio, sr = librosa.load(filename, sr = 16000, offset=start, 
        duration=duration)
	return audio

def _load_audio(item):
    filename = item['audiofilename']
    item['audio'] = {}
    item['audio']['array'] = load_audio(filename)
    item['audio']['sampling_rate'] = 16000
    return item
    
def load_vocab(vocab_filename = None):
    if not vocab_filename: return locations.vocab_sampa
    with open(vocab_filename) as fin:
        vocab = json.load(fin)
    return vocab
