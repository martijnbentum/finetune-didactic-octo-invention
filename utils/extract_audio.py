import librosa
import numpy as np
import soundfile as sf

def load_audio(filename, sample_rate = 16_000, mono = True,
    start = None, end = None, duration = None):
    if start != None and end != None:
        duration = end - start
    if start == None: start = 0.0
    audio, sr = librosa.load(filename, sr = sample_rate, mono = mono,
        offset = start, duration = duration)
    return audio

def write_audio_to_wav(audio, filename, sample_rate = 16_000, mono = True):
    if audio.shape[0] == 2: audio = to_mono(audio)
    sf.write(filename, audio, sample_rate, subtype = 'PCM_16')

def to_mono(audio):
    return librosa.to_mono(audio)

def samples_to_seconds(audio, sample_rate = 16_000):
    return len(audio) / sample_rate

def audio_to_duration(audio, sample_rate = 16_000):
    return samples_to_seconds(audio, sample_rate)

def seconds_to_samples(seconds, sample_rate = 16_000):
    return seconds / sample_rate

def seconds_to_index(seconds, sample_rate = 16_000):
    return int(round(seconds_to_samples(seconds, sample_rate)))

def start_time_duration_to_indices(start_time, duration, sample_rate = 16_000):
    start_index = seconds_to_index(start_time, sample_rate)
    end_index = seconds_to_index(start_time + duration, sample_rate)
    return start_index, end_index

def make_silence(duration, sample_rate = 16_000):
    silence = np.zeros(round(duration * sample_rate))
    return silence


def combine_audios(audios, pause = .15, sample_rate = 16_000):
    silence = make_silence(pause, sample_rate)
    o = []
    for i, audio in enumerate(audios):
        if i == len(audios) -1: o.append(audio)
        else: o.extend([audio,silence])
    return np.concatenate((o))
