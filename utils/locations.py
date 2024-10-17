import json

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
vocab_file = cache_dir + 'vocab.json'
cgn_phrases_dir = '../cgn_phrases/'
cgn_phrases_dict = '../cgn_wav_filename_phrases.json'
cgn_speaker_dict = '../cgn_speakers.json'

with open(vocab_file) as fin:
    vocab = json.load(fin)
