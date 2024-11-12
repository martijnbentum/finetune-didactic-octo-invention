import json
import os
from pathlib import Path
from . import locations


names= ['facebook_300m/','dutch_test_best/', 'random_model/',
    'dutch_test_25k/', 'dutch_test_5k/', 'music_100k', 'audio_non_speech_100k',
    'dutch_960_1/', 'dutch_960_4/','dutch_960_10/', 'dutch_960_100/',
    'dutch_960_1000/', 'dutch_960_10000/', 'dutch_960_2000/',
    'dutch_960_100000/'] 

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)

def make_results_dict(names = names, save = False):
    d = {}
    names = sorted(names)
    for name in names:
        ort= f'../orthographic_{name}'
        d[name] = {}
        d[name]['orthographic'] = get_wer_and_step(ort)
        sampa = f'../sampa_{name}'
        d[name]['sampa'] = get_wer_and_step(sampa)
    save_json(d, '../wer_results.json')
    return d

def load_trainer_state(path):
    if not path: return
    with open(path / 'trainer_state.json') as f:
        d = json.load(f)
    return d

def get_wer_and_step(directory):
    if not os.path.exists(str(directory)):
        print(f'{directory} does not exist')
        return
    path = Path(locations.make_checkpoint_path(directory))
    d = load_trainer_state(path)
    if not d: 
        print(f'No trainer_state.json in {directory}')
        return
    output = []
    for log in d['log_history']:
        if 'eval_wer' in log:
            output.append([log['eval_wer'], log['step']])
    return output
    

