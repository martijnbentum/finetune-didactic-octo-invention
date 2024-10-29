import json
from pathlib import Path

names= ['facebook_300m/','dutch_test_best/',
    'dutch_test_25k/', 'dutch_test_5k/']

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)

def make_results_dict(names = names, save = False):
    d = {}
    for name in names:
        ort= f'../orthographic_{name}'
        d[name] = {}
        d[name]['orthographic'] = get_wer_and_step(ort)
        sampa = f'../sampa_{name}'
        d[name]['sampa'] = get_wer_and_step(sampa)
    save_json(d, '../wer_results.json')
    return d

def load_trainer_state(path):
    with open(path / 'trainer_state.json') as f:
        d = json.load(f)
    return d

def get_wer_and_step(directory):
    path = make_path(directory)
    d = load_trainer_state(path)
    output = []
    for log in d['log_history']:
        if 'eval_wer' in log:
            output.append([log['eval_wer'], log['step']])
    return output
    

def make_path(directory_name):
    p = Path(directory_name) 
    checkpoint = get_latest_created_dir(p)
    return checkpoint


def get_latest_created_dir(path):
    directories = [p for p in Path(path).iterdir() if p.is_dir()]
    if not directories:
        return None  # Return None if there are no directories
    latest_dir = max(directories, key=lambda p: p.stat().st_ctime)
    return latest_dir