import jiwer
import json
import glob
from matplotlib import cm
import matplotlib.colors as mcolors
import os
from pathlib import Path
import random
import re
import shutil
from . import locations


names= ['xlsr_300m/', 'random_model/',
    'music', 'audio_non_speech',
    'dutch_960_1/', 'dutch_960_4/','dutch_960_10/', 'dutch_960_100/',
    'dutch_960_1000/', 'dutch_960_2000/', 'dutch_960_10000/', 
    'dutch_960_25000/', 'dutch_960_50000/', 'dutch_960_100000/'] 

def _filter_and_order_dutch(directories):
    dutch = []
    for directory, checkpoint, wers in directories:
        if 'dutch' in directory:
            dutch.append([directory, checkpoint, wers])
    dutch = sorted(dutch, key = lambda x: int(x[0].split('_')[-1].strip('/')))
    return dutch

def _filter_non_speech(directories):
    non_speech = []
    for directory, checkpoint, wers in directories:
        for term in ['audio_non_speech', 'music', 'random']:
            if term in directory:
                non_speech.append([directory, checkpoint, wers])
                break
    return non_speech

def _filter_other(directories):
    other = []
    for directory, checkpoint, wers in directories:
        for term in ['base', 'xlsr']:
            if term in directory:
                other.append([directory, checkpoint, wers])
                break
    return other

def find_type(directory):
    if 'dutch' in directory:
        return 'dutch', 'speech'
    for term in ['audio_non_speech', 'music', 'random']:
        if term in directory:
            return term, 'non speech'
    if 'base' in directory:
        return 'english', 'speech'
    if 'xlsr' in directory:
        return 'multilingual', 'speech'
    raise ValueError(f'Could not find type for {directory}')

    


def directory_to_name(directory, remove_terms = []):
    name = directory.split('/')[-2]
    for term in remove_terms:
        name = name.replace(term, '')
    name = name.replace('_', ' ')
    name = name.replace('960', '')
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
    for directory, checkpoint, wers in directories:
        name = directory_to_name(directory, remove_terms)
        output.append([directory, checkpoint, wers, name])
    return output

class Result():
    def __init__(self, directory, checkpoint):
        self.directory = directory
        self.checkpoint = checkpoint
        self.term, self.type = find_type(directory)
        self.finetune_wers = get_wer_and_step(directory)
        self.transcription = 'sampa' if 'sampa' in directory else 'orthographic'
        self.name = directory_to_name(directory, remove_terms = [
            self.transcription])
        self.wer_cer_dict = checkpoint_to_wer_cer_dict(self.checkpoint, 
            self.transcription)
        self.color = None

    def __repr__(self):
        m = f'{self.name} | {self.transcription}' 
        if self.best_eval_wer:
            m += f' | eval wer: {self.best_eval_wer:.2f}'
        if self.wer:
            m += f' | test wer: {self.wer:.2f}'
        if self.cer:
            metric = 'cer' if self.transcription == 'orthographic' else 'per'
            m += f' | {metric}: {self.wer:.2f}'
        return m

    def set_color(self, color):
        self.color = color

    @property
    def best_eval_wer(self):
        if not self.finetune_wers: return None
        return min([x[0] for x in self.finetune_wers])

    @property
    def cer(self):
        if not self.wer_cer_dict: return None
        return self.wer_cer_dict['cer']

    @property
    def wer(self):
        if not self.wer_cer_dict: return None
        return self.wer_cer_dict['wer']

    def sample_transcription(self, n = 1):
        if not self.wer_cer_dict: return []
        return random.sample(self.wer_cer_dict['data'], n)

    @property
    def sample_character_alignment(self):
        if not self.wer_cer_dict: return []
        line = self.sample_transcription(1)[0]
        print(line['character_alignment'])

    @property
    def sample_word_alignment(self):
        if not self.wer_cer_dict: return []
        line = self.sample_transcription(1)[0]
        print(line['word_aligment'])

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

def get_wer_and_step(directory = '', checkpoint_directory = None):
    if not checkpoint_directory:
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
    reference = line['sentence'].lower()
    hypothesis = line['hyp']
    co = jiwer.process_characters(reference, hypothesis)
    wo = jiwer.process_words(reference, hypothesis)
    line['word_aligment'] = jiwer.visualize_alignment(wo)
    line['character_alignment'] = jiwer.visualize_alignment(co)
    line['wer'] = wo.wer
    line['cer'] = co.cer

def handle_ref_hyp_file(filename, overwrite = False, verbose = True):
    output_filename = filename.replace('.json', '_wer.json')
    p = Path(output_filename)
    if p.exists() and not overwrite:
        if verbose:print(f'{output_filename} exists, loading it')
        with open(p) as f:
            d = json.load(f)
        return d
    elif p.exists() and overwrite: print(f'Overwriting {output_filename}')
    if verbose:print(f'Handling {filename}')
    with open(filename) as f:
        temp = json.load(f)
    if 'data' not in temp:d = {'data':temp}
    else: d = temp
    for line in d['data']:
        _handle_ref_hyp_line(line)
    hyp = [x['hyp'] for x in d['data']]
    ref = [x['sentence'].lower() for x in d['data']]
    d['wer'] = jiwer.wer(ref, hyp)
    d['cer'] = jiwer.cer(ref, hyp)
    if verbose:print(f'Saving to {output_filename}')
    with open(output_filename, 'w') as f:
        json.dump(d, f)
    return d

def finetuned_checkpoint_to_wer(checkpoint, test_name = 'o', 
    transcription = None, overwrite = False, verbose = True):
    filename = checkpoint_to_test_transcription_filename(checkpoint,
        test_name = test_name, transcription = transcription)
    if filename.exists(): 
        if verbose: print('handling', filename)
        return handle_ref_hyp_file(str(filename), overwrite = overwrite,
            verbose = verbose)
    m = f'{filename} does not exist, doing nothing\n'
    if verbose:
        m += f'use test_transcribe to create it first\n'
        m += f'test_transcribe.handle_test_set({checkpoint},' 
        m += f' component = {test_name}, transcription = {transcription},'
        m += f' save = True, overwrite = False, device = -1)'
    print(m)

def finetuned_checkpoints_to_wer_dicts(checkpoints, test_name = 'o',
    transcription = None, overwrite = False, verbose = False):
    output = []
    for cp in checkpoints:
        o = finetuned_checkpoint_to_wer(cp, test_name = test_name,
            transcription = transcription, overwrite = overwrite,
            verbose = verbose)
        if not o: continue
        name = checkpoint_to_name(cp)
        print(cp, name)
        try: step = int(name.split('_pt-')[-1].split('_')[0])
        except ValueError: step = None
        wer = o['wer']
        version = checkpoint_to_model_version(cp, test_name)
        result_line = f'version: {version.ljust(21)}, step: {step}'
        result_line += f', test: {test_name}, wer: {wer:.2f}'
        temp = {'checkpoint': cp, 'name': name, 'version':version, 
            'test': test_name, 'wer': wer, 'step':step, 'results': o, 
            'result_line': result_line}
        output.append(temp)
    return output

def checkpoint_to_name(checkpoint):
    p = Path(checkpoint)
    name = p.parent.stem
    return name
        
def checkpoint_to_model_version(checkpoint, test_name = 'o',):
    if 'huibert_the_first' in checkpoint:
        return 'huibert_the_first'
    if 'huibert_the_second' in checkpoint:
        return 'huibert_the_second'
    if 'wav2vec2_the_first' in checkpoint:
        return 'wav2vec2_the_first'
    if 'wav2vec2_the_second' in checkpoint:
        return 'wav2vec2_the_second'
    name = checkpoint_to_name(checkpoint)
    return name.replace('_ft-'+test_name, '')
        

def handle_old_finetuned_directories():
    fn = locations.sampa_finetuned_directories()
    fn += locations.orthographic_finetuned_directories()
    checkpoints = [x[1] for x in fn]
    output = []
    for checkpoint in checkpoints:
        o = finetuned_checkpoint_to_wer(checkpoint)
        if o: output.append(o)
    return output

def checkpoint_to_test_transcription_filename(checkpoint, test_name = 'o',
    transcription = None):
    if transcription is None:
        if 'sampa' in checkpoint:
            transcription = 'sampa' 
        elif 'orthographic' in checkpoint:
            transcription = 'orthographic'
        else:
            m = 'Could not determine transcription from directory,'
            m += ' please provide it explicitly'
            raise ValueError(m)
    filename = Path(checkpoint) / f'{test_name}_test_{transcription}_hyp.json'
    return filename

def copy_all_ref_hyp_wer_files_to_goal_dir(
    goal_dir = '../all_ref_hyp_wer_files/'):
    fn = locations.sampa_finetuned_directories()
    fn += locations.orthographic_finetuned_directories()
    for directory, checkpoint in fn:
        name = directory_to_name(directory).replace(' ', '_')
        transcription = 'sampa' if 'sampa' in directory else 'orthographic'
        filename = f'o_test_{transcription}_hyp_wer.json'
        input_filename = Path(checkpoint) / filename
        output_filename = Path(goal_dir) / f'{name}_{filename}'
        print(f'copying {filename} to {output_filename}')
        shutil.copyfile(input_filename, output_filename)

def checkpoint_to_wer_cer_dict(checkpoint, transcription):
    filename = Path(checkpoint) / f'o_test_{transcription}_hyp_wer.json'
    if not filename.exists(): 
        print(f'{filename} does not exist')
        return None
    with open(filename) as f:
        d= json.load(f)
    return d


def collect_wer_cer():
    fn = locations.sampa_finetuned_directories()
    fn += locations.orthographic_finetuned_directories()
    d = []
    for directory, checkpoint in fn:
        transcription = 'sampa' if 'sampa' in directory else 'orthographic'
        data = checkpoint_to_wer_cer_dict(checkpoint, transcription)
        name = directory_to_name(directory)
        print(name, directory)
        order = int(name.split(' ')[-1]) if 'dutch' in name else 0
        if 'xlsr' in name: order = 10**6
        if 'base' in name: order = 10**6 - 1
        short_name = 'dutch' if 'dutch' in name else name.split(' ')[-1]
        if 'xlsr' in name: short_name = 'xlsr'
        d.append({'wer': data['wer'], 'cer': data['cer'], 
            'transcription': transcription, 'order': order, 
            'short_name': short_name,'name': name})
    return d



