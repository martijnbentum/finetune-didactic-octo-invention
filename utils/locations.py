import json
import glob
from pathlib import Path

cgn_dir = '/vol/bigdata/corpora2/CGN2/'
cgn_audio_dir = cgn_dir + 'data/audio/wav/'
cgn_annot_dir = cgn_dir + 'data/annot/text/'
cgn_phoneme_dir = cgn_annot_dir + 'fon/'
cgn_ort_dir = cgn_annot_dir + 'ort/'
cgn_awd_dir = cgn_annot_dir + 'awd/'
cgn_fon_dir = cgn_annot_dir + 'fon/'
cgn_speaker_file = cgn_dir + 'data/meta/text/speakers.txt'
local_awd = '../awd/'
local_fon= '../fon/'
local_ort= '../ort/'


cache_dir = '../WAV2VEC_DATA/'
json_dir = '../JSON/'
vocab_sampa_dir= '../vocab_sampa/'
vocab_sampa_file = vocab_sampa_dir + 'vocab.json'
vocab_orthographic_dir= '../vocab_orthographic/'
vocab_orthographic_file = vocab_orthographic_dir + 'vocab.json'
cgn_phrases_dir = '../cgn_phrases/'
cgn_phrases_dict = '../cgn_wav_filename_phrases.json'
cgn_speaker_dict = '../cgn_speakers_7.json'

helper_files_directory = 'helper_files/'

with open(vocab_sampa_file) as fin:
    vocab_sampa = json.load(fin)
with open(vocab_orthographic_file) as fin:
    vocab_orthographic = json.load(fin)

def _finetuned_directories(transcription = 'orthographic',
    add_latest_checkpoint = True):
    directories = glob.glob(f'../{transcription}_*/')
    output = []
    for d in directories:
        fn = glob.glob(d + 'checkpoint-*')
        if fn: 
            if add_latest_checkpoint:
                output.append([d, make_checkpoint_path(d)])
            else: output.append(d)
        else:print(f'No checkpoint in {d}, {fn}, {d+ "checkpoint-*"}')
    return output

def orthographic_finetuned_directories(add_latest_checkpoint = True):
    return _finetuned_directories('orthographic', add_latest_checkpoint)

def sampa_finetuned_directories(add_latest_checkpoint = True):
    return _finetuned_directories('sampa', add_latest_checkpoint)

def make_checkpoint_path(directory_name):
    p = Path(directory_name) 
    checkpoint = get_latest_created_dir(p)
    return checkpoint

def get_latest_created_dir(path):
    directories = [p for p in Path(path).iterdir() if p.is_dir()]
    if not directories:
        return None  # Return None if there are no directories
    latest_dir = max(directories, key=lambda p: p.stat().st_ctime)
    latest_dir = latest_dir.resolve().as_posix()
    if not latest_dir.endswith('/'): latest_dir = latest_dir + '/'
    return latest_dir

def beeld_en_geluid_directories():
    base_ft_cv = ['orthographic_dutch_55k_400k_ft-cv', 
        '/vol/mlusers/mbentum/beg/models/base-40min-ft/']
    base_ft_cgn_o =['orthographic_dutch_55k_400k_ft-cgn-o', 
        '/vol/mlusers/mbentum/beg/models/base-40min-ft/']
