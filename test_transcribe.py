import json
import transcribe
from utils import locations
from utils import clean_text
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
        

def handle_ifadv(check_point, save = True, overwrite = False, device = -1):
    pipeline = load_pipeline(check_point, device = device)
    d = json.load(open('../JSON/ifadv_phrases.json'))
    filename = Path(check_point)
    filename = filename / f'ifadv_test_hyp.json'
    if filename.exists() and save and not overwrite:
        print('File exists, doing nothing', filename)
        return
    for item in progressbar(d['data']):
        transcribe.transcribe_ifadv_item(item, pipeline)
    if save:
        with open(filename,'w') as fout:
            json.dump(d, fout)
        print('Saved to', filename)
    del pipeline
    return d

def load_mls_sentences(split = 'test'):
    with open('/vol/mlusers/mbentum/mls/dutch_mls_sentences_zs.tsv') as fin:
        lines = fin.read().split('\n')
    header, temp= lines[0], lines[1:]
    data = []
    for line in temp:
        d = {k:v for k,v in zip(header.split('\t'), line.split('\t'))}
        for h in ['start_time','end_time', 'duration']:
            d[h] = float(d[h])
        d['sentence'] = d['text'].strip('"')
        d['in_pretraining'] = d['in_pretraining'] == 'True'
        if split == 'all': pass
        elif split == 'in_pretraining' and d['in_pretraining']: continue
        elif split in ['train','test','dev']:
            if d['split'] != split: continue
        data.append(d)
    return data

def handle_mls(check_point, save = True, overwrite = False, device = -1,
    split = 'test'):
    pipeline = load_pipeline(check_point, device = device)
    d = load_mls_sentences(split)
    filename = Path(check_point)
    filename = filename / f'mls_test_hyp.json'
    if filename.exists() and save and not overwrite:
        print('File exists, doing nothing', filename)
        return
    directory = Path('/vol/mlusers/mbentum/mls/dutch/audio/')
    for line in progressbar(d):
        f = line['audio_filename']
        audio_filename = directory / line['split'] / f
        line['hyp'] = apply_pipeline(pipeline, str(audio_filename))
    if save:
        with open(filename,'w') as fout:
            json.dump(d, fout)
        print('Saved to', filename)
    del pipeline
    return d

def load_cv_sentences(split = 'test'):
    if split != 'test': raise ValueError('split should be test')
    f = '/vol/tensusers/mbentum/INDEEP/LD/COMMON_VOICE_DUTCH/test.tsv'
    with open(f) as fin:
        lines = fin.read().split('\n')
    data = []
    for line in lines[1:]:
        if not line: continue
        d = {}
        line = line.split('\t')
        d['id'], d['audio_filename'], d['sentence'] = line[:3]
        d['sentence'] = clean_text.normalize_text(d['sentence'])
        data.append(d)
    return data


def handle_cv(check_point, save = True, overwrite = False, device = -1,
    split = 'test'):
    pipeline = load_pipeline(check_point, device = device)
    d = load_cv_sentences(split)
    filename = Path(check_point)
    filename = filename / f'cv_test_hyp.json'
    if filename.exists() and save and not overwrite:
        print('File exists, doing nothing', filename)
        return
    directory=Path('/vol/tensusers/mbentum/INDEEP/LD/COMMON_VOICE_DUTCH/clips/')
    for line in progressbar(d):
        f = line['audio_filename']
        audio_filename = directory / f
        line['hyp'] = apply_pipeline(pipeline, str(audio_filename))
    if save:
        with open(filename,'w') as fout:
            json.dump(d, fout)
        print('Saved to', filename)
    del pipeline
    return d

def test_speech_training_interspeech_article_models_on_ifadv(device = 1, 
    overwrite = False):
    d, o = locations.path_and_names_for_speech_training_article()
    names = ['fb-en','dutch_base','fb-voxp-100k','nonspeech']
    for name in names:
        print('Testing', name)
        cp = locations.make_checkpoint_path(o[name])
        handle_ifadv(cp, save = True, overwrite = overwrite, device = device)

def test_speech_training_interspeech_article_models_on_o(device =1, 
    overwrite = False):
    d, o = locations.path_and_names_for_speech_training_article()
    names = ['fb-en','dutch_base','fb-voxp-100k','nonspeech']
    for name in names:
        print('Testing', name)
        cp = locations.make_checkpoint_path(o[name])
        handle_test_set(cp, component = 'o', transcription = 'orthographic',
            save = True, overwrite = overwrite, device = device)
        
def test_speech_training_interspeech_article_models_on_mls(device = 1,
    overwrite = False, split = 'test'):
    d, o = locations.path_and_names_for_speech_training_article()
    names = ['fb-en','dutch_base','fb-voxp-100k','nonspeech']
    for name in names:
        print('Testing', name)
        cp = locations.make_checkpoint_path(o[name])
        handle_mls(cp, save = True, overwrite = overwrite, 
        device = device, split = split)


def test_speech_training_interspeech_article_models_on_cv(device = 1,
    overwrite = False, split = 'test'):
    d, o = locations.path_and_names_for_speech_training_article()
    names = ['fb-en','dutch_base','fb-voxp-100k','nonspeech']
    for name in names:
        print('Testing', name)
        cp = locations.make_checkpoint_path(o[name])
        handle_cv(cp, save = True, overwrite = overwrite, 
        device = device, split = split)


