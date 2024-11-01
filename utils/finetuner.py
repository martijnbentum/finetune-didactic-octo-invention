from decouple import config
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

