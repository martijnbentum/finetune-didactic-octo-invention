from decouple import config
import json
import os
from pathlib import Path
from utils import wav2vec2_model
from utils import locations

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
    model = wav2vec2_model.load_model(checkpoint_dir, 
        transcription = transcription)
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

def finetune_dutch_large_bg_orthographic(directory = '', warmup_steps = 5000,
    learning_rate = 5e-5, per_device_train_batch_size = 33, 
    num_train_epochs = 122, dataset_name = 'o',
    eval_steps = 1000, save_steps = 1000, experiment_name = ''):
    if not directory:
        directory = '/vol/mlusers/mbentum/beg/models/large-40min/'
    if not experiment_name:
        experiment_name='/vol/mlusers/mbentum/beg/models/large_dutch_ft_comp-o-ft122/'
    vocab_filename = f'{directory}vocab.json'
    model, vocab, processor = load_large_model(directory)
    trainer = wav2vec2_model.load_trainer(dataset_name, 
        transcription ='orthographic', 
        experiment_name=experiment_name, vocab_filename = vocab_filename,
        processor = processor, model = model, warmup_steps = warmup_steps,
        learning_rate = learning_rate,
        per_device_train_batch_size = per_device_train_batch_size,
        num_train_epochs = num_train_epochs, save_steps = save_steps,
        eval_steps = eval_steps)
    assert len(vocab) == model.config.vocab_size
    return model, vocab, trainer
    
def finetune_dutch_base_bg_orthographic_new(directory = '', warmup_steps = 1000,
    learning_rate = 3e-4, per_device_train_batch_size = 50,
    num_train_epochs = 140, eval_steps = 1000, save_steps = 1000,
    group_by_length = False):
    if not directory:
        directory = '/vol/mlusers/mbentum/beg/models/base-40min/'
    experiment_name='/vol/mlusers/mbentum/beg/models/base_dutch_ft_comp-o-ft140/'
    vocab_filename = f'{directory}vocab.json'
    model, vocab, processor = load_large_model(directory)
    trainer = wav2vec2_model.load_trainer('o', transcription ='orthographic', 
        experiment_name=experiment_name, vocab_filename = vocab_filename,
        processor = processor, model = model, warmup_steps = warmup_steps,
        learning_rate = learning_rate, num_train_epochs = num_train_epochs,
        per_device_train_batch_size = per_device_train_batch_size,
        save_steps = save_steps, eval_steps = eval_steps, 
        group_by_length = group_by_length)
    assert len(vocab) == model.config.vocab_size
    return model, vocab, trainer

def finetune_xls_r_orthographic(warmup_steps = 5000, learning_rate=3e-4,
    num_train_epochs=122,
    eval_steps = 1000, save_steps = 1000, experiment_name = ''):
    if not experiment_name:
        experiment_name='/vol/mlusers/mbentum/beg/models/xls_r_300m_ft_comp-o-ft122/'
    vocab_filename = '/vol/mlusers/mbentum/beg/models/large-40min/vocab.json'
    vocab = json.load(open(vocab_filename))
    tokenizer = wav2vec2_model.Wav2Vec2CTCTokenizer(vocab_filename)
    feature_extractor = wav2vec2_model.load_feature_extractor()
    processor = wav2vec2_model.Wav2Vec2Processor(
        feature_extractor =feature_extractor, tokenizer = tokenizer)
    wav2vec2_model.processor = processor
    model = wav2vec2_model.load_model(processor = wav2vec2_model.processor)
    trainer = wav2vec2_model.load_trainer('o', transcription ='orthographic', 
        experiment_name=experiment_name, vocab_filename = vocab_filename,
        processor = processor, model = model, warmup_steps = warmup_steps,
        learning_rate = learning_rate, num_train_epochs = num_train_epochs,
        save_steps = save_steps, eval_steps = eval_steps)
    assert len(vocab) == model.config.vocab_size
    print('saving to ', experiment_name)
    return model, vocab, trainer
    

def finetune_xlsr_orthographic(warmup_steps = 5000, learning_rate=5e-5,
    num_train_epochs=122):
    experiment_name='/vol/mlusers/mbentum/beg/models/xlsr_ft_comp-o/'
    vocab_filename = '/vol/mlusers/mbentum/beg/models/large-40min/vocab.json'
    vocab = json.load(open(vocab_filename))
    tokenizer = wav2vec2_model.wav2vec2ctctokenizer(vocab_filename)
    feature_extractor = wav2vec2_model.load_feature_extractor()
    processor = wav2vec2_model.wav2vec2processor(
        feature_extractor =feature_extractor, tokenizer = tokenizer)
    wav2vec2_model.processor = processor
    name = 'facebook/wav2vec2-large-xlsr-53'
    model = wav2vec2_model.load_model(name,processor = wav2vec2_model.processor)
    trainer = wav2vec2_model.load_trainer('o', transcription ='orthographic', 
        experiment_name=experiment_name, vocab_filename = vocab_filename,
        processor = processor, model = model, warmup_steps = warmup_steps,
        learning_rate = learning_rate, num_train_epochs = num_train_epochs)
    assert len(vocab) == model.config.vocab_size
    print('saving to ', experiment_name)
    return model, vocab, trainer
    

def finetune_dutch_xls_r_large_bg_orthographic(directory = '', 
    warmup_steps = 5000,
    learning_rate = 3e-4, per_device_train_batch_size = 50, 
    num_train_epochs = 122, dataset_name = 'o',
    eval_steps = 1000, save_steps = 1000, experiment_name = ''):
    if not directory:
        directory = '/vol/mlusers/mbentum/beg/models/xls-r-300m/'
    if not experiment_name:
        experiment_name='/vol/mlusers/mbentum/beg/models/xls_r_large_dutch_ft_comp-o-ft122_fl/'
    vocab_filename = f'{directory}vocab.json'
    model, vocab, processor = load_large_model(directory)
    trainer = wav2vec2_model.load_trainer(dataset_name, 
        transcription ='orthographic', 
        experiment_name=experiment_name, vocab_filename = vocab_filename,
        processor = processor, model = model, warmup_steps = warmup_steps,
        learning_rate = learning_rate,
        per_device_train_batch_size = per_device_train_batch_size,
        num_train_epochs = num_train_epochs, save_steps = save_steps,
        eval_steps = eval_steps)
    assert len(vocab) == model.config.vocab_size
    return model, vocab, trainer

def finetune_for_speech_training(name):
    names = ['nonspeech', 'fb-en', 'fb-xlsr-53', 'gronlp-nl-base',
        'fb-voxp-100k', 'fb-voxp-nl']
    if name not in names: 
        print(f'{name} not in {names}, doing nothing')
        return
    wav2vec2_model.processor = None
    wav2vec2_model.tokenizer = None
    cp_dir, exp_dir = locations.path_and_names_for_speech_training_article()
    checkpoint_dir = cp_dir[name]
    experiment_name = exp_dir[name]
    model = wav2vec2_model.load_model(checkpoint_dir, transcription = 'orthographic')
    trainer = wav2vec2_model.load_trainer('o', 'orthographic', 
        experiment_name, save_steps = 1000, eval_steps = 1000, model = model)
    p =  Path(experiment_name)
    print(experiment_name, p.exists(), list(p.glob('*')))
    return model, trainer

def finetune_fb_en_orthographic(warmup_steps = 5000, learning_rate=5e-5,
    num_train_epochs= 60, per_device_train_batch_size =50):
    experiment_name='/vol/mlusers/mbentum/speech_training/models/'
    experiment_name+='wav2vec2_base-fb-en-fto/'
    vocab_filename = '/vol/mlusers/mbentum/beg/models/large-40min/vocab.json'
    vocab = json.load(open(vocab_filename))
    tokenizer = wav2vec2_model.Wav2Vec2CTCTokenizer(vocab_filename)
    feature_extractor = wav2vec2_model.load_feature_extractor()
    processor = wav2vec2_model.Wav2Vec2Processor(
        feature_extractor =feature_extractor, tokenizer = tokenizer)
    wav2vec2_model.processor = processor
    name = 'facebook/wav2vec2-base'
    model = wav2vec2_model.load_model(name,processor = wav2vec2_model.processor)
    trainer = wav2vec2_model.load_trainer('o', transcription ='orthographic', 
        experiment_name=experiment_name, vocab_filename = vocab_filename,
        processor = processor, model = model, warmup_steps = warmup_steps,
        learning_rate = learning_rate, num_train_epochs = num_train_epochs,
        per_device_train_batch_size = per_device_train_batch_size)
    assert len(vocab) == model.config.vocab_size
    print('saving to ', experiment_name)
    return model, vocab, trainer

def finetune_dutch_base_orthographic(warmup_steps = 5000, learning_rate=5e-5,
    num_train_epochs= 60, per_device_train_batch_size =50):
    experiment_name='/vol/mlusers/mbentum/speech_training/models/'
    experiment_name+='wav2vec2_base-dutch-fto/'
    vocab_filename = '/vol/mlusers/mbentum/beg/models/large-40min/vocab.json'
    vocab = json.load(open(vocab_filename))
    tokenizer = wav2vec2_model.Wav2Vec2CTCTokenizer(vocab_filename)
    feature_extractor = wav2vec2_model.load_feature_extractor()
    processor = wav2vec2_model.Wav2Vec2Processor(
        feature_extractor =feature_extractor, tokenizer = tokenizer)
    wav2vec2_model.processor = processor
    name = '../w2v_pretrain_checkpoints/dutch_960/checkpoint_229_100000'
    model = wav2vec2_model.load_model(name,processor = wav2vec2_model.processor)
    trainer = wav2vec2_model.load_trainer('o', transcription ='orthographic', 
        experiment_name=experiment_name, vocab_filename = vocab_filename,
        processor = processor, model = model, warmup_steps = warmup_steps,
        learning_rate = learning_rate, num_train_epochs = num_train_epochs,
        per_device_train_batch_size = per_device_train_batch_size)
    assert len(vocab) == model.config.vocab_size
    print('saving to ', experiment_name)
    return model, vocab, trainer

def finetune_non_speech_orthographic(warmup_steps = 5000, learning_rate=5e-5,
    num_train_epochs= 60, per_device_train_batch_size =50):
    experiment_name='/vol/mlusers/mbentum/speech_training/models/'
    experiment_name+='wav2vec2_non_speech-fto/'
    vocab_filename = '/vol/mlusers/mbentum/beg/models/large-40min/vocab.json'
    vocab = json.load(open(vocab_filename))
    tokenizer = wav2vec2_model.Wav2Vec2CTCTokenizer(vocab_filename)
    feature_extractor = wav2vec2_model.load_feature_extractor()
    processor = wav2vec2_model.Wav2Vec2Processor(
        feature_extractor =feature_extractor, tokenizer = tokenizer)
    wav2vec2_model.processor = processor
    name = '/vol/mlusers/mbentum/speech_training/models/nonspeech_model'
    model = wav2vec2_model.load_model(name,processor = wav2vec2_model.processor)
    trainer = wav2vec2_model.load_trainer('o', transcription ='orthographic', 
        experiment_name=experiment_name, vocab_filename = vocab_filename,
        processor = processor, model = model, warmup_steps = warmup_steps,
        learning_rate = learning_rate, num_train_epochs = num_train_epochs,
        per_device_train_batch_size = per_device_train_batch_size)
    assert len(vocab) == model.config.vocab_size
    print('saving to ', experiment_name)
    return model, vocab, trainer

def finetune_dutch_base_bg_nik_orthographic(directory = ''):
    if not directory:
        directory = '/vol/mlusers/mbentum/beg/models/base-40min/'
    experiment_name='/vol/mlusers/mbentum/beg/models/base_dutch_ft_comp-o-80ksteps/'
    vocab_filename = f'{directory}vocab.json'
    model, vocab, processor = load_large_model(directory)
    trainer = wav2vec2_model.load_base_bg_trainer(model, vocab, processor,
        experiment_name, vocab_filename)
    assert len(vocab) == model.config.vocab_size
    return model, vocab, trainer
