from .wav2vec2_data import load_cgn_dataset
from .wav2vec2_data import DataCollatorCTCWithPadding 
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments
from transformers import Trainer
from transformers import TrainerCallback, TrainerState, TrainerControl
import numpy as np
from datetime import datetime
import evaluate
import os
import json
from utils import locations

wer_metric = evaluate.load('wer')

class freezeLogicCallback(TrainerCallback):
    def __init__(self, model: Wav2Vec2ForCTC):
        self.model = model

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.model.freeze_base_model()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.global_step == 10_000:
            for param in self.model.wav2vec2.parameters():
                param.requires_grad = True

            self.model.freeze_feature_encoder()

def load_model(directory = '' ):
    if not directory:
        directory = '/vol/mlusers/mbentum/beg/models/base-40min/'
    processor = Wav2Vec2Processor.from_pretrained(directory)
    model = Wav2Vec2ForCTC.from_pretrained(directory)
    model.freeze_base_model()
    return model

def load_base_bg_trainer(model,vocab, processor, experiment_name,
    vocab_filename):
    if not os.path.isdir(experiment_name):os.mkdir(experiment_name)
    data_collator = load_data_collator(transcription = 'orthographic',
        vocab_file = vocab_filename, processor = processor)
    datasets= preprocess_cgn_dataset('o', 
        transcription = 'orthographic', maximum_length = None,
        processor = processor)
    training_args = training_arguments_base(experiment_name)
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=datasets['train'],
        eval_dataset=datasets['dev'],
        tokenizer=processor.feature_extractor,
        callbacks=[freezeLogicCallback(model)],

    )
    return trainer
