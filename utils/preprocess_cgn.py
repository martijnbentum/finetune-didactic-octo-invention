import copy
import json
from matplotlib import pyplot as plt
import numpy as np
import os
from pathlib import Path
from utils import extract_audio
from utils import locations
from progressbar import progressbar

def load_speakers():
    '''load the json file with all non-overlapping phrases from all speakers'''
    with open(locations.cgn_speaker_dict) as fin:
        speaker_dict = json.load(fin)
    return speaker_dict

def exclude(speakers = None,exclude = ['c','d','m'], minimum_length = 1):
    '''exclude phrases from components in exclude list and with duration < 1
    seconds.
    '''
    if not speakers: speakers = load_speakers()
    speakers = exclude_componenents(speakers, exclude)
    speakers = exclude_short_phrases(speakers, minimum_length, False)
    return speakers


def write_audio_cgn_phrases(speakers = None):
    '''write wav files for each non overlapping phrase for each speaker.
    by default it uses phrases generate by exclude, excluding materials from
    components c d & m and phrases with duration < 1 seconds.
    '''
    if not speakers: speakers = exclude()
    filenames = {}
    for speaker in progressbar(speakers.values()):
        for phrase in speaker['phrases']:
            filename = phrase_to_filename(phrase)
            filename = Path(locations.cgn_phrases_dir) / filename
            audio = load_phrase_audio(phrase)
            extract_audio.write_audio_to_wav(audio, filename)
            filenames[filename.as_posix()] = phrase
    save_json(filenames, locations.cgn_phrases_dict)
    return filenames

def save_json(data, filename):
    '''save data to filename as json'''
    with open(filename, 'w') as fout:
        json.dump(data, fout)


def phrase_to_filename(phrase):
    '''creates a unique filename based on phrase information
    does not include a path
    '''
    speaker = phrase['speaker_id']
    cgn_id = phrase['cgn_id']
    start_end = str(phrase['start_time']), str(phrase['end_time'])
    start_end = '-'.join([x.replace('.','__') for x in start_end])
    filename = '_'.join([speaker, cgn_id, start_end]) + '.wav'
    return filename

def load_phrase_audio(phrase):
    '''loads audio for a single phrase in an np array.'''
    filename = Path(locations.cgn_dir) / phrase['audio_filename']
    if not filename.exists(): return
    start, end = phrase['start_time'], phrase['end_time']
    audio = extract_audio.load_audio(filename, start = start, end = end)
    return audio

def exclude_componenents(speakers, exclude = ['c','d','m']):
    '''exclude all phrases from specific components from the spoken dutch corpus
    if a speaker only occurs in these components the speaker is removed as well
    default exclude components:
        c & d (telephone conversations, different sample rate 8 KHz)
        m (sermons very poor audio quality)
    '''
    info(speakers)
    print('excluding components:', ', '.join(exclude))
    new_speakers = {}
    for k, speaker in speakers.items():
        new_phrases = []
        for phrase in speaker['phrases']:
            if phrase['component'] in exclude: continue
            new_phrases.append(phrase)
        if len(new_phrases) == 0: continue
        new_speaker = copy.copy(speaker)
        new_speaker['phrases'] = new_phrases
        new_speaker['duration'] = sum([p['duration'] for p in new_phrases])
        new_speakers[k] = new_speaker
    info(new_speakers)
    return new_speakers

def exclude_short_phrases(speakers, minimum_length = 1, start_info = True):
    '''excluding all phrases shorter than minimum_length (default 1 seconds).
    if no phrases remain for a speaker the speaker is removed.
    '''
    if start_info: info(speakers)
    print('excluding phrases shorter than', minimum_length, 'seconds')
    new_speakers = {}
    for k, speaker in speakers.items():
        new_phrases = []
        for phrase in speaker['phrases']:
            if phrase['duration'] < minimum_length: continue
            new_phrases.append(phrase)
        if len(new_phrases) == 0: continue
        new_speaker = copy.copy(speaker)
        new_speaker['phrases'] = new_phrases
        new_speaker['duration'] = sum([p['duration'] for p in new_phrases])
        new_speakers[k] = new_speaker
    info(new_speakers)
    return new_speakers

def info(speakers):
    '''show number of phrases and total duration'''
    n_phrases = 0
    duration = 0
    n_speakers = len(speakers)
    for speaker in speakers.values():
        n_phrases += len(speaker['phrases'])
        duration += speaker['duration']
    print('# speakers:', n_speakers,'# phrases:',n_phrases,'duration:',
        round(duration/3600),'hours')
    _extra_info(speakers)

def _hist_durations(durations, color = 'black', alpha = 1, new_figure = True,
    show = True, label = '', add_legend = False, bins = 100):
    plt.ion()
    if new_figure: plt.figure()
    plt.hist(durations, color = color,alpha = alpha, bins = bins, label = label)
    plt.grid(alpha=.3)
    plt.ylabel('counts')
    plt.xlabel('phrase duration in seconds')
    if add_legend: plt.legend()
    if show: plt.show()


def plot_phrase_duration_histogram(speakers):
    durations = []
    for speaker in speakers.values():
        durations.extend([p['duration'] for p in speaker['phrases']])
    _hist_durations(durations, show = True, label = 'raw phrases')

def compute_duration_per_speaker(speakers):
    durations = []
    for speaker in speakers.values():
        durations.append(speaker['duration'])
    return durations

def plot_duration_per_speaker(speakers, bins = 10):
    durations = compute_duration_per_speaker(speakers)
    _hist_durations(durations, bins = bins)
    plt.xlabel('speech materials per speaker in seconds')
    plt.show()
    plt.xlim(0,5000)
    plt.show()
    print('nspeakers:', len(durations))
    print('min - max:',np.min(durations),np.max(durations))
    print('mean', np.mean(durations))
    print('median',np.median(durations))

        
