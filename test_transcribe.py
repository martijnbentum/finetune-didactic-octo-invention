import json
import transcribe
from utils import locations
from progressbar import progressbar
from pathlib import Path

def load_test_set(component = 'o', transcription = 'sampa'):
    filename = locations.json_dir 
    filename += f'{component}_test_{transcription}.json'
    with open(filename) as fin:
        d = json.load(fin)
    return d

def load_pipeline(recognizer_dir = None, device = -1):
    if recognizer_dir is None: 
        recognizer_dir = '../orthographic_dutch_960_100000/checkpoint-13671/'
    pipeline = transcribe.load_pipeline(recognizer_dir = recognizer_dir,
        device = device)
    return pipeline

def apply_pipeline(pipeline, audio_filename):
    o = pipeline(audio_filename)
    return o['text']

def handle_test_set(recognizer_dir, component = 'o', transcription = 'sampa',
    save = False, overwrite = False, device = -1):
    filename = Path(recognizer_dir)
    filename = filename / f'{component}_test_{transcription}_hyp.json'
    if filename.exists() and save and not overwrite:
        print('File exists, doing nothing', filename)
        return
    d = load_test_set(component = component, transcription = transcription)
    pipeline = load_pipeline(recognizer_dir, device = device)
    for line in progressbar(d['data']):
        audio_filename = line['audiofilename']
        line['hyp'] = apply_pipeline(pipeline, audio_filename)
    if save:
        with open(filename,'w') as fout:
            json.dump(d, fout)
        print('Saved to', filename)
    del pipeline
    return d

def test_finetuned_models_set(component = 'o', transcription = 'sampa',
    device = -1, overwrite = False):
    if transcription == 'sampa':
        directories = locations.sampa_finetuned_directories()
    elif transcription == 'orthographic':
        directories = locations.orthographic_finetuned_directories()
    else: raise ValueError('transcription should be sampa or orthographic',
        transcription)
    for directory, checkpoint in directories:
        print('Testing', directory)
        print('finetuned model checkpoint:', checkpoint)
        handle_test_set(checkpoint, component = component,
            transcription = transcription, save = True, overwrite = overwrite,
            device = device)
        

