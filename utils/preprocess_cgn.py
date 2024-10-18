import copy
import json
from matplotlib import pyplot as plt
import numpy as np
import os
from scripts import extract_audio
from src.speech_training import paths
from progressbar import progressbar

def load_speakers():
    '''load the json file with all non-overlapping phrases from all speakers'''
    return json.load(open(paths.cgn_speakers))

def exclude(speakers = None,exclude = ['c','d','m'], minimum_length = 2):
    '''exclude phrases from components in exclude list and with duration < 2
    seconds.
    '''
    if not speakers: speakers = load_speakers()
    speakers = exclude_componenents(speakers, exclude)
    speakers = exclude_short_phrases(speakers, minimum_length, False)
    return speakers

def combine_phrases(speakers, minimum_length = 8, maximum_length = 15,
    group_audio_files = False):
    '''combine phrases of a given speaker with a duration in range
    minimum_length, maximum_length.
    group_audio_files can be used to only combine audio files from the
    same recording
    the combined and single phrases are stored under the keys
    combine_phrases, single_phrases
    '''
    if not speakers: speakers = exclude()
    new_speakers = {}
    for k, speaker in speakers.items():
        new_speaker = handle_combine_phrases_speaker(speaker,
            minimum_length, maximum_length,group_audio_files)
        new_speakers[k] = new_speaker
    return new_speakers

def write_audio_cgn_phrases(speakers = None):
    '''write wav files for each non overlapping phrase for each speaker.
    by default it uses phrases generate by exclude, excluding materials from
    components c d & m and phrases with duration < 2 seconds.
    '''
    if not speakers: speakers = exclude()
    filenames = {}
    for speaker in progressbar(speakers.values()):
        for phrase in speaker['phrases']:
            filename = phrase_to_filename(phrase)
            filename = paths.CGN_PHRASES / filename
            audio = load_phrase_audio(phrase)
            extract_audio.write_audio_to_wav(audio, filename)
            filenames[filename] = phrase
    return filenames

def write_audio_cgn_combined_phrases(speakers = None):
    '''write wav files for each non overlapping phrase for each speaker.
    by default it uses phrases generate by exclude, excluding materials from
    components c d & m and phrases with duration < 2 seconds.

    It combines phrases with duration < 8 duration into phrases of <= 15
    seconds
    '''
    if not speakers:
        speakers = exclude()
        speakers = combine_phrases(speakers)
    filenames = {}
    for speaker in progressbar(speakers.values()):
        for phrase in speaker['single_phrases']:
            filename = phrase_to_filename(phrase)
            filename = paths.CGN_COMBINED_PHRASES / filename
            audio = load_phrase_audio(phrase)
            extract_audio.write_audio_to_wav(audio, filename)
            filenames[filename] = phrase
        for index, phrase in enumerate(speaker['combined_phrases']):
            filename = combined_phrase_to_filename(phrase,index)
            filename = paths.CGN_COMBINED_PHRASES / filename
            audio = load_combined_phrase_audio(phrase)
            extract_audio.write_audio_to_wav(audio, filename)
            filenames[filename] = phrase
    return filenames

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

def combined_phrase_to_filename(combined_phrase, index):
    '''creates a filename with speaker filename ids and an index of the
    combined phrase for the given speaker
    '''
    speaker = combined_phrase['phrases'][0]['speaker_id']
    index = 'cp-index-' + str(index)
    cgn_ids = '-'.join([x['cgn_id'] for x in combined_phrase['phrases']])
    filename = '_'.join([speaker, index, cgn_ids]) + '.wav'
    return filename

def load_phrase_audio(phrase):
    '''loads audio for a single phrase in an np array.'''
    filename = paths.CGN / phrase['audio_filename']
    if not filename.exists(): return
    start, end = phrase['start_time'], phrase['end_time']
    audio = extract_audio.load_audio(filename, start = start, end = end)
    return audio

def load_combined_phrase_audio(combined_phrase, pause = .15):
    '''loads audio for multiple phrases into a single np array with
    silences between the audio, silence duration in seconds
    can be set with pause
    '''
    audios = []
    for phrase in combined_phrase['phrases']:
        audio = load_phrase_audio(phrase)
        audios.append(audio)
    audio = extract_audio.combine_audios(audios, pause)
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

def exclude_short_phrases(speakers, minimum_length = 2, start_info = True):
    '''excluding all phrases shorter than minimum_length (default 2 seconds).
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


def _group_phrases_according_to_audio_file(speaker):
    audio_files = {}
    for phrase in speaker['phrases']:
        filename = phrase['audio_filename']
        if filename not in audio_files.keys():
            audio_files[filename] = []
        audio_files[filename].append(phrase)
    return audio_files

def handle_combine_phrases_speaker(speaker, minimum_length, maximum_length,
    group_audio_files = False):
    '''combines phrases from the same audio from a given speaker.
    the combined phrases duration are in the range minimum_length,
    maximum_length
    phrases to long to combine
    '''
    if group_audio_files:
        audio_files = _group_phrases_according_to_audio_file(speaker)
        combined_phrases, single_phrases = [], []
        for phrases in audio_file.values():
            cp, sp = phrases_to_combined_phrases(phrases, minimum_length,
                maximum_length)
            combined_phrases.extend(cp)
            single_phrases.extend(sp)
    else: combined_phrases, single_phrases = phrases_to_combined_phrases(
        speaker['phrases'], minimum_length, maximum_length)
    new_speaker = copy.copy(speaker)
    new_speaker['single_phrases'] = single_phrases
    new_speaker['combined_phrases'] = combined_phrases
    return new_speaker

def _new_combined_phrase():
    '''create a new combine_phrase dictionary
    '''
    combined_phrase = {'duration':0, 'phrases': []}
    return combined_phrase

def _add_phrase(combined_phrase, phrase):
    '''add a phrase to the combined_phrase dict and update the duration value
    '''
    combined_phrase['phrases'].append(phrase)
    combined_phrase['duration'] += phrase['duration']

def _handle_add_combined_phrase(single_phrases, combined_phrases,
    combined_phrase):
    '''add combined_phrase to combined phrases list if it contains
    more than one phrase
    if there are no phrases ignore it
    if it contains 1 phrase add the phrase to the single_phrases list
    '''
    phrases = combined_phrase['phrases']
    if len(phrases) == 0: pass
    elif len(phrases) == 1: single_phrases.extend(combined_phrase['phrases'])
    else: combined_phrases.append(combined_phrase)

def phrases_to_combined_phrases(phrases, minimum_length, maximum_length):
    '''combine phrase into combined_phrases with a length in range
    minimum_length - maximum_length
    phrases should be a list of phrases that can be combined i.e.,
    from the same speaker and/or same audio file
    returns a list of combined phrases and single phrases
    '''
    combined_phrases, single_phrases = [], []
    combined_phrase = _new_combined_phrase()
    for i,phrase in enumerate(phrases):
        duration = combined_phrase['duration']
        if phrase['duration'] >= maximum_length:
            single_phrases.append(phrase)
        elif duration < minimum_length:
            if duration + phrase['duration'] > maximum_length:
                single_phrases.append(phrase)
            else: _add_phrase(combined_phrase, phrase)
        else:
            if duration + phrase['duration'] >= maximum_length:
                _handle_add_combined_phrase(
                    single_phrases, combined_phrases, combined_phrase)
                combined_phrase = _new_combined_phrase()
            _add_phrase(combined_phrase, phrase)
        if i == len(phrases) -1:
            # ensure last combined phrase is added
            _handle_add_combined_phrase(
                single_phrases, combined_phrases, combined_phrase)
    return combined_phrases, single_phrases

def _combined_phrases_available(speakers):
    speaker= next(iter(speakers.values()))
    return 'single_phrases' in speaker.keys()

def _extra_info(speakers):
    '''shows extra information for combined phrases'''
    if not _combined_phrases_available(speakers):return
    n_single_phrases = 0
    single_phrases_duration = 0
    n_combined_phrases = 0
    combined_phrases_duration = 0
    for speaker in speakers.values():
        sp, cp = speaker['single_phrases'], speaker['combined_phrases']
        n_single_phrases += len(sp)
        n_combined_phrases += len(cp)
        single_phrases_duration += sum([p['duration'] for p in sp])
        combined_phrases_duration += sum([p['duration'] for p in cp])
    print('# single phrases:',n_single_phrases,
        'duration:', round(single_phrases_duration / 3600), 'hours',
        '# combined_phrases:', n_combined_phrases,
        'duration:', round(combined_phrases_duration / 3600), 'hours')


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

def _plot_extra_histogram(speakers):
    if not _combined_phrases_available(speakers):return
    durations = []
    for speaker in speakers.values():
        durations.extend([p['duration'] for p in speaker['single_phrases']])
        durations.extend([p['duration'] for p in speaker['combined_phrases']])
    _hist_durations(durations, 'orange', .7, False, label = 'combined phrases',
        add_legend = True)

def plot_phrase_duration_histogram(speakers):
    durations = []
    for speaker in speakers.values():
        durations.extend([p['duration'] for p in speaker['phrases']])
    show = False if _combined_phrases_available(speakers) else True
    _hist_durations(durations, show = show, label = 'raw phrases')
    _plot_extra_histogram(speakers)

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

        
