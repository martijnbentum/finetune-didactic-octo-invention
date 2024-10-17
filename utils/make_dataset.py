from utils import locations
from utils import number2word
from pathlib import Path
import unicodedata
import json
import numpy as np
import os
import re

def save_json(filename, d, overwrite = False):
    if os.path.exists(filename) and not overwrite:
        raise ValueError('File already exists', filename)
    with open(filename, 'w') as fout:
        json.dump(d, fout, indent = 4)
    

def make_dataset(name = 'default', selected_data = None, 
    sentence_field = 'sampa', save = False, overwrite = False):
    if not selected_data: selected_data = select_component('o')
    train, dev, test = make_train_dev_test_split(selected_data)
    output = []
    for d,split in zip([train,dev,test],['train','dev','test']):
        sf = sentence_field
        filename = locations.json_dir + name + '_' + split + '_' + sf + '.json'
        dd = phrases_dict_to_dataset(d, sentence_field)
        if save: save_json(filename, dd, overwrite)
        output.append( d )
    train, dev, test = output
    return train, dev, test

def phrases_dict_to_dataset(phrases_dict, sentence_field = 'sampa'):
    '''converst a dictionary of phrase items to dataset format'''
    output = {'data':[]}
    for key, item in phrases_dict.items():
        dataset_item = phrases_item_to_dataset_item(item, key, sentence_field)
        output['data'].append(dataset_item)
    return output

def phrases_item_to_dataset_item(item, key, sentence_field = 'sampa'):
    '''convert a single phrase item to a dataset item'''
    if sentence_field == 'sampa': f = clean_sampa
    elif sentence_field == 'orthographic': f = clean_orthographic
    else: raise ValueError('sentence_field should be sampa or orthographic')
    output = {}
    output['sentence'] = f(item[sentence_field])
    filename = locations.cgn_phrases_dir + key.split('/')[-1]
    filename = Path(filename).resolve()
    if not filename.exists():
        raise ValueError('File does not exist', filename)
    output['audiofilename'] = filename.as_posix()
    return output


def load_cgn_phrases_dict():
    '''load json file with CGN phrases'''
    with open(locations.cgn_phrases_dict) as fin:
        d = json.load(fin)
    return d

def select_component(component = 'o', phrases_dict = None):
    '''select phrases with from a  specific component of the CGN'''
    if phrases_dict is None: phrases_dict = load_cgn_phrases_dict()
    d = {}
    for k,v in phrases_dict.items():
        if v['component'] == component:
            d[k] = v
    return d

def select_components(components = ['o','k'], phrases_dict = None):
    '''select phrases with from multiple components of the CGN'''
    if phrases_dict is None: phrases_dict = load_cgn_phrases_dict()
    d = {}
    for k,v in phrases_dict.items():
        if v['component'] in components:
            d[k] = v
    return d


def _select_items_from_dict(d, keys):
    '''select items from a dictionary based on a list of keys'''
    output = {}
    for k in keys:
        output[k] = d[k]
    return output

def to_duration(d):
    '''calculate the total duration of a dictionary of phrases'''
    duration = 0
    for value in d.values():
        duration += value['duration']
    return duration

def to_text(d, field_name = 'sampa'):
    '''convert a dictionary of phrases to a single string
    field_name: sampa or orthographic; for phoneme or orthographic transcription
    '''
    output = []
    for value in d.values():
        output.append(value[field_name])
    return ' '.join(output)

def to_character_set(d, field_name = 'sampa'):
    '''convert a dictionary of phrases to a set of characters used in the
    transcription
    '''
    text = to_text(d,field_name)
    return set(text)

def find_examples(d, field_name = 'sampa', character = 'a'):
    '''find examples in a dictionary of phrases that contain a specific
    character
    '''
    output = []
    for k,v in d.items():
        if character in v[field_name]:
            output.append(v)
    return output

def make_train_dev_test_split(selected_data = None):
    '''make a train, dev, test split for a specific component of the CGN
    '''
    np.random.seed(42)
    if not selected_data: selected_data = select_component('o')
    d = selected_data
    keys = list(d.keys())
    np.random.shuffle(keys)
    train_index, dev_index, test_index = make_train_dev_test_indices(len(d))
    train = _select_items_from_dict(d,keys[:train_index])
    dev = _select_items_from_dict(d,keys[train_index:dev_index])
    test = _select_items_from_dict(d,keys[dev_index:test_index])
    return train, dev, test

def make_train_dev_test_indices(n, train_size = 0.8, dev_size = 0.1, 
    test_size = 0.1):
    '''make indices for train, dev, and test sets'''
    assert abs(train_size + dev_size + test_size - 1) < 0.0001
    train_index = int(n*train_size)
    dev_index = int(n*(train_size + dev_size))
    test_index = n + 1
    return train_index, dev_index, test_index

def clean_sampa(sampa):
    '''remove ! and ë ö to e o respectively 
    map multiple spaces to a single space
    '''
    sampa = sampa.replace('!','')
    sampa = sampa.replace('ë','e')
    sampa = sampa.replace('ö','o')
    sampa = re.sub(r'\s+',' ',sampa)
    return sampa

def clean_orthographic(ort):
    '''remove punctuation and diacritics from orthographic transcription
    map multiple spaces to a single space    
    map all characters to lowercase
    '''
    ort = number2word.map_all_numbers_to_words(ort)
    ort = ort.lower()
    ort = ort.replace(" 't ", ' het ')
    ort = ort.replace(" 'k ", ' ik ')
    ort = ort.replace(" 'm ", ' hem ')
    ort = ort.replace("'ns", 'eens')
    ort = ort.replace("z'n", 'zijn')
    ort = ort.replace("m'n", 'mijn')
    ort = ort.replace("d'r", 'der')
    ort = ort.replace("da's", 'dat is')
    ort = ort.replace("ggg", ' ')
    ort = ort.replace("xxx", ' ')
    ort = ort.replace('!',' ')
    ort = ort.replace('?',' ')
    ort = ort.replace('.',' ')
    ort = ort.replace('_',' ')
    ort = ort.replace('-',' ')
    ort = ort.replace('&',' ')
    ort = ort.replace("'",' ')
    ort = ort.replace('ø','o')
    ort = re.sub(r'\*.', ' ', ort)
    ort = remove_diacritics(ort)
    ort = re.sub(r'\s+',' ',ort)
    return ort

def remove_diacritics(text):
    # Normalize the text to decompose characters
    normalized_text = unicodedata.normalize('NFD', text)
    # Filter out characters that are diacritics
    return ''.join([c for c in normalized_text if not unicodedata.combining(c)])

