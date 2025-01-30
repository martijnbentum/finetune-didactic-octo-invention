from .wav2vec2_data import load_cgn_dataset
from .wav2vec2_data import DataCollatorCTCWithPadding 
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments
from transformers import Trainer
import numpy as np
from datetime import datetime
import evaluate
import os
import json
from utils import locations

processor = None
tokenizer = None
wer_metric = evaluate.load('wer')


def load_vocab(vocab_filename = None):
    if not vocab_filename: return locations.vocab_sampa
    with open(vocab_filename) as fin:
        vocab = json.load(fin)
    return vocab

def load_tokenizer(vocab_file= locations.vocab_sampa_file):
    tokenizer = Wav2Vec2CTCTokenizer(vocab_file,
        unk_token='[UNK]',pad_token='[PAD]',
        word_delemiter_token='|')
    return tokenizer

def load_feature_extractor():
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1,
        sampling_rate=16000, padding_value=0.0, do_normalize=True,
        return_attention_mask=True)
    return feature_extractor

def load_processor(vocab_file = locations.vocab_sampa_file):
    global processor
    if processor: return processor
    global tokenizer
    if not tokenizer:
        tokenizer = load_tokenizer(vocab_file)
    feature_extractor = load_feature_extractor()
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor,
        tokenizer=tokenizer)
    return processor

def preprocess_item(item):
    audio = item['audio']

    item['input_values'] = processor(audio['array'],
        sampling_rate = audio['sampling_rate']).input_values[0]
    item['input_length'] = len(item['input_values'])

    with processor.as_target_processor():
        item['labels'] = processor(item['sentence']).input_ids
    return item

def _preprocess_datasets(datasets,maximum_length = None, sampling_rate = 16000):
    d = datasets
    for key in d.keys():
        column_names = d[key].column_names
        d[key] = d[key].map(preprocess_item, remove_columns= column_names)
        if maximum_length:
            maximum = maximum_length * sampling_rate
            d[key] = d[key].filter(lambda x: x < maximum,
                input_columns=['input_length'])
    return d

def preprocess_cgn_dataset(dataset_name, transcription = 'sampa', 
    maximum_length = None, vocab_file= None, processor = None):
    if not vocab_file and not processor:
        if transcription == 'sampa': vocab_file = locations.vocab_sampa_file
        elif transcription == 'orthographic': 
            vocab_file = locations.vocab_orthographic_file
        else: raise ValueError('transcription should be sampa or orthographic')
    if not processor:
        load_processor(vocab_file = vocab_file)
    d = load_cgn_dataset(dataset_name,transcription)
    d = _preprocess_datasets(d, maximum_length = maximum_length)
    return d

def load_data_collator(transcription = 'sampa', processor = None, 
    vocab_file = None):
    if not processor:
        if not vocab_file:
            if transcription == 'sampa': vocab_file = locations.vocab_sampa_file
            elif transcription == 'orthographic': 
                vocab_file = locations.vocab_orthographic_file
        processor = load_processor(vocab_file = vocab_file)
    return DataCollatorCTCWithPadding(processor = processor,padding = True)

def compute_metrics(pred):
    processor = load_processor()
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis = -1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    save_preds_references(pred_str,label_str,wer)
    return {"wer": wer}

def save_preds_references(preds,references,wer):
    wer = str(int(wer * 100))
    d = datetime.now().strftime("%d_%m_%Y_%H_%M")
    filename = locations.cache_dir + 'log_dev_wer_' + wer + '-'+d
    output = []
    for pred, ref in zip(preds,references):
        output.append(pred + '\t' + ref)
    with open(filename,'w') as fout:
        fout.write('\n'.join(output))


def make_config_for_random_model(transcription = 'sampa'):
    from transformers import Wav2Vec2Config
    if transcription == 'sampa': vocab_file = locations.vocab_sampa_file
    elif transcription == 'orthographic': 
        vocab_file = locations.vocab_orthographic_file
    else: raise ValueError('transcription should be sampa or orthographic')
    processor = load_processor(vocab_file)
    config = Wav2Vec2Config(
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        hidden_size=768,        # Hidden size of the model
        num_attention_heads=12, # Number of attention heads
        num_hidden_layers=12,   # Number of transformer layers
        intermediate_size=3072, # Intermediate size in the feedforward layer
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        cache_dir = locations.cache_dir
    )
    return config 

def load_random_model(config = None, transcription = 'sampa'):
    '''loads a randomly initialized model
    optionally takes a model as a template and returns a 
    randomly initialized model
    '''
    if not config: config = make_config_for_random_model(transcription)
    model = Wav2Vec2ForCTC(config)
    model.freeze_feature_extractor()
    return model

def load_model(model_name = "facebook/wav2vec2-xls-r-300m", processor = None,
    transcription = 'sampa'):
    if transcription == 'sampa': vocab_file = locations.vocab_sampa_file
    elif transcription == 'orthographic': 
        vocab_file = locations.vocab_orthographic_file
    else: raise ValueError('transcription should be sampa or orthographic')
    if not processor: processor = load_processor(vocab_file)
    model = Wav2Vec2ForCTC.from_pretrained(
        model_name,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        cache_dir = locations.cache_dir
    )
    model.freeze_feature_extractor()
    return model

def load_training_arguments(experiment_name, num_train_epochs = 21,
    warmup_steps = 300, learning_rate = 3e-4, 
    per_device_train_batch_size = 33, eval_steps =300, save_steps = 300):
    if not os.path.isdir(experiment_name):os.mkdir(experiment_name)
    training_args = TrainingArguments(
        output_dir=experiment_name,
        group_by_length=True,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        num_train_epochs=num_train_epochs,
        gradient_checkpointing=True,
        fp16=True,
        save_steps=save_steps,
        eval_steps=eval_steps,
        logging_steps=50,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,#1000,#300,
        save_total_limit=3,
        push_to_hub=False,
    )
    return training_args

def load_trainer(dataset_name, transcription, experiment_name,model = None, 
    training_args = None, maximum_length = None, 
    datasets = None,train = 'train',evaluate='dev', num_train_epochs = 21,
    processor = None, vocab_filename = None, warmup_steps = 300,
    learning_rate = 3e-4, per_device_train_batch_size = 33,
    eval_steps = 300, save_steps = 300):
    # experiment_name = comp_name + '_' + experiment_name
    print('set processor')
    if not vocab_filename:
        if transcription == 'sampa': vocab_filename = locations.vocab_sampa_file
        elif transcription == 'orthographic': 
            vocab_filename = locations.vocab_orthographic_file
        else: raise ValueError('transcription should be sampa or orthographic')
    if not processor: processor = load_processor(vocab_file = vocab_filename)
    print('make data collator')
    data_collator = load_data_collator(transcription = transcription,
        vocab_file = vocab_filename, processor = processor)
    if not model:
        print('load model')
        model = load_model(transcription = transcription)
    if not training_args:
        print('load training arguments')
        training_args = load_training_arguments(experiment_name, 
            num_train_epochs = num_train_epochs, warmup_steps = warmup_steps,
            learning_rate = learning_rate, 
            per_device_train_batch_size = per_device_train_batch_size,
            eval_steps = eval_steps, save_steps = save_steps)
    if not datasets:
        print('load datasets')
        datasets= preprocess_cgn_dataset(dataset_name, 
            transcription = transcription, maximum_length = maximum_length,
            processor = processor)
    print('defining the trainer')
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=datasets[train],
        eval_dataset=datasets[evaluate],
        tokenizer=processor.feature_extractor,
    )
    return trainer

def do_component_training(dataset_name,transcription, experiment_name):
    trainer = load_trainer(dataset_name, transcription, experiment_name)
    trainer.train()
    return trainer
