from decouple import config
import json
import os
from pathlib import Path
from utils import wav2vec2_model

def check_pretrained_checkpoint_exists(checkpoint_name,directory = None ):
    if not directory:
        directory = '../w2v_pretrain_checkpoints/dutch_960/'
    p = Path(directory)
    filenames = list(p.glob(f'*_{checkpoint_name}.pt'))
    if len(filenames) == 0:
        return False
    if len(filenames) > 1:
        print(filenames)
        raise ValueError('Multiple checkpoints found')
    print(f'Checkpoint {checkpoint_name} found: {filenames[0]}')
    return True

def download_dutch_pretrained_checkpoint(name, output_directory = None):
    if not output_directory:
        output_directory = '../w2v_pretrain_checkpoints/dutch_960/'
    cmd = config('scp_snel')
    cmd += config('dutch_pretrain_dir')
    if check_pretrained_checkpoint_exists(name):
        print(f'Checkpoint {name} already exists, doing nothing.')
        return
    if name == 'best':
        checkpoint_name = 'checkpoint_best.pt'
    else:
        checkpoint_name = f'checkpoint_*_{name}.pt'
    cmd += checkpoint_name + ' ' + output_directory 
    print(f'Downloading checkpoint... \n{cmd}')
    os.system(cmd)
    return cmd

def download_dutch_pretrained_checkpoints():
    checkpoints = [1,2,10,100,1000,10_000,100_000]
    for checkpoint in checkpoints:
        _ = download_dutch_pretrained_checkpoint(checkpoint)



def collect_pretrained_checkpoints(pretrained_model_dir = None):
    if not pretrained_model_dir:
        pretrained_model_dir = '../w2v_pretrain_checkpoints/dutch_960/'
    p = Path(pretrained_model_dir)
    dirs = p.glob('*/pytorch_model.bin')
    return [str(d.parent) for d in dirs]


def finetune_pretrained_checkpoint(checkpoint_dir, experiment_name ,
    dataset_name = 'o', transcription = 'orthographic'):
    model = wav2vec2_model.load_model(checkpoint_dir)
    trainer = wav2vec2_model.load_trainer(dataset_name, transcription, 
        experiment_name)
    trainer.train()
     
    
    

def handle_pretrained_model(name, checkpoints):
    pass

def load_large_model(directory = '' ):
    if not directory:
        directory = '/vol/mlusers/mbentum/beg/models/large-40min/'
    vocab_filename = f'{directory}vocab.json'
    vocab = json.load(open(vocab_filename))
    tokenizer = wav2vec2_model.Wav2Vec2CTCTokenizer(vocab_filename)
    feature_extractor = wav2vec2_model.load_feature_extractor()
    processor = wav2vec2_model.Wav2Vec2Processor(
        feature_extractor =feature_extractor, tokenizer = tokenizer)
    wav2vec2_model.processor = processor
    model = wav2vec2_model.load_model(directory, processor = processor)
    return model, vocab, processor

def finetune_dutch_large_bg_orthographic(directory = ''):
    if not directory:
        directory = '/vol/mlusers/mbentum/beg/models/large-40min/'
    experiment_name='/vol/mlusers/mbentum/beg/models/large_dutch_ft_comp-o/'
    vocab_filename = f'{directory}vocab.json'
    model, vocab, processor = load_large_model(directory)
    trainer = wav2vec2_model.load_trainer('o', transcription ='orthographic', 
        experiment_name=experiment_name, vocab_filename = vocab_filename,
        processor = processor, model = model)
    assert len(vocab) == model.config.vocab_size
    return model, vocab, trainer
    
def finetune_dutch_base_bg_orthographic(directory = '', warmup_steps = 2000,
    learning_rate = 3e-4, per_device_train_batch_size = 33):
    if not directory:
        directory = '/vol/mlusers/mbentum/beg/models/base-40min/'
    experiment_name='/vol/mlusers/mbentum/beg/models/base_dutch_ft_comp-o/'
    vocab_filename = f'{directory}vocab.json'
    model, vocab, processor = load_large_model(directory)
    trainer = wav2vec2_model.load_trainer('o', transcription ='orthographic', 
        experiment_name=experiment_name, vocab_filename = vocab_filename,
        processor = processor, model = model, warmup_steps = warmup_steps,
        learning_rate = learning_rate, 
        per_device_train_batch_size = per_device_train_batch_size)
    assert len(vocab) == model.config.vocab_size
    return model, vocab, trainer
    
