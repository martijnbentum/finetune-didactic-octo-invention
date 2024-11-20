import jiwer
import json
import glob
from matplotlib import cm
import matplotlib.colors as mcolors
import os
from pathlib import Path
import re
from . import locations


names= ['xlsr_300m/', 'random_model/',
    'music', 'audio_non_speech',
    'dutch_960_1/', 'dutch_960_4/','dutch_960_10/', 'dutch_960_100/',
    'dutch_960_1000/', 'dutch_960_2000/', 'dutch_960_10000/', 
    'dutch_960_25000/', 'dutch_960_50000/', 'dutch_960_100000/'] 

def _filter_and_order_dutch(directories):
    dutch = []
    for directory, checkpoint in directories:
        if 'dutch' in directory:
            dutch.append([directory, checkpoint])
    dutch = sorted(dutch, key = lambda x: int(x[0].split('_')[-1].strip('/')))
    return dutch

def _filter_non_speech(directories):
    non_speech = []
    for directory, checkpoint in directories:
        for term in ['audio_non_speech', 'music', 'random']:
            if term in directory:
                non_speech.append([directory, checkpoint])
                break
    return non_speech

def _filter_other(directories):
    other = []
    for directory, checkpoint in directories:
        for term in ['base', 'xlsr']:
            if term in directory:
                other.append([directory, checkpoint])
                break
    return other

def directory_to_name(directory, remove_terms = []):
    name = directory.split('/')[-2]
    for term in remove_terms:
        name = name.replace(term, '')
    name = name.replace('_', ' ')
    name= re.sub(r'\s+', ' ', name)
    name = name.strip()
    return name

def add_colors_to_directories(directories, gradient = False, index = 0):
    if gradient:
        color_names = cm.get_cmap('viridis',len(directories))
    else:
        color_names = list(mcolors.BASE_COLORS.keys())
    for line in directories:
        if gradient: color_name = color_names(index)
        else: color_name = color_names[index]
        line.append(color_name)
        index += 1
    return directories

def _add_names(directories, remove_terms = []):
    output = []
    for directory, checkpoint in directories:
        name = directory_to_name(directory, remove_terms)
        output.append([directory, checkpoint, name])
    return output

class Results:
    def __init__(self):
        self._set_info()

    def _set_info(self):
        self.index = 0
        d = locations.sampa_finetuned_directories()
        self._sampa_directories = _add_wer_and_step(d)
        d = locations.orthographic_finetuned_directories()
        self._orthographic_directories = _add_wer_and_step(d)
        self._set_dutch()
        self._set_non_speech()
        self._set_other()
        self.sampa = self.sampa_dutch + self.non_speech_sampa 
        self.sampa += self.other_sampa
        self.orthographic = self.orthographic_dutch 
        self.orthographic += self.non_speech_orthographic
        self.orthographic += self.other_orthographic
        self.all = self.sampa + self.orthographic

    def _set_dutch(self):
        d = _filter_and_order_dutch(self._sampa_directories)
        d = _add_names(d,['sampa','960'])
        self.sampa_dutch = add_colors_to_directories(d, gradient = True)
        d = _filter_and_order_dutch(self._orthographic_directories)
        d = _add_names(d, ['orthographic', '960'])
        self.orthographic_dutch = add_colors_to_directories(d,
            gradient = True) 

    def _set_non_speech(self):
        d = _filter_non_speech(self._sampa_directories)
        d = _add_names(d)
        self.non_speech_sampa = add_colors_to_directories(d, index = self.index)
        d =  _filter_non_speech(self._orthographic_directories)
        d = _add_names(d)
        self.non_speech_orthographic = add_colors_to_directories(d, 
            index = self.index)
        self.index += len(d)
            
    def _set_other(self):
        d = _filter_other(self._sampa_directories)
        d = _add_names(d)
        self.other_sampa = add_colors_to_directories(d, index = self.index)
        d = _filter_other(self._orthographic_directories)
        d = _add_names(d)
        self.other_orthographic = add_colors_to_directories(d, 
            index = self.index)
        self.index += len(d)

            
        
    


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

def _add_wer_and_step(directories):
    output = []
    for directory, checkpoint in directories:
        wers = get_wer_and_step(directory)
        if wers:
            output.append([directory, checkpoint, wers])
    return output

def get_wer_and_step(directory):
    if not os.path.exists(str(directory)):
        print(f'{directory} does not exist')
        return
    checkpoint_directory = locations.make_checkpoint_path(directory)
    if not checkpoint_directory:
        print(f'{checkpoint_directory} does not exist')
        return
    path = Path(checkpoint_directory)
    d = load_trainer_state(path)
    if not d: 
        print(f'No trainer_state.json in {directory}')
        return
    output = []
    for log in d['log_history']:
        if 'eval_wer' in log:
            output.append([log['eval_wer'], log['step']])
    return output
    
def _handle_ref_hyp_line(line):
    reference = line['sentence']
    hypothesis = line['hyp']
    co = jiwer.process_characters(reference, hypothesis)
    wo = jiwer.process_words(reference, hypothesis)
    line['word_aligment'] = jiwer.visualize_alignment(wo)
    line['character_alignment'] = jiwer.visualize_alignment(co)
    line['wer'] = wo.wer
    line['cer'] = co.cer

def handle_ref_hyp_file(filename):
    with open(filename) as f:
        d = json.load(f)
    for line in d['data']:
        _handle_ref_hyp_line(line)
    hyp = [x['hyp'] for x in d['data']]
    ref = [x['sentence'] for x in d['data']]
    d['wer'] = jiwer.wer(ref, hyp)
    d['cer'] = jiwer.cer(ref, hyp)
    output_filename = filename.replace('.json', '_wer.json')
    print(f'Saving to {output_filename}')
    with open(output_filename, 'w') as f:
        json.dump(d, f)

def handle_all_ref_hyp_files():
    fn = locations.sampa_finetuned_directories()
    fn += locations.orthographic_finetuned_directories()
    for directory, checkpoint in fn:
        transcription = 'sampa' if 'sampa' in directory else 'orthographic'
        filename = Path(checkpoint) / f'o_test_{transcription}_hyp.json'
        print('handling', filename)
        if filename.exists():
            handle_ref_hyp_file(str(filename))

def collect_wer_cer():
    fn = locations.sampa_finetuned_directories()
    fn += locations.orthographic_finetuned_directories()
    d = {}
    for directory, checkpoint in fn:
        transcription = 'sampa' if 'sampa' in directory else 'orthographic'
        filename = Path(checkpoint) / f'o_test_{transcription}_hyp_wer.json'
        name = directory.split('/')[-2]
        print(directory,name)
        if not filename.exists(): 
            print(f'{filename} does not exist')
            continue
        with open(filename) as f:
            data = json.load(f)
        d[name] ={'wer': data['wer'], 'cer': data['cer']}
    return d



