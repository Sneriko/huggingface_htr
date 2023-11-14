from HTRDataset import HTRDataset
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from PIL import Image
import jiwer
from transformers import TrOCRProcessor
from torch.utils.data import ConcatDataset
from pathlib import Path
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader
from transformers import VisionEncoderDecoderModel
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import evaluate
from transformers import default_data_collator
import os


class CustomRandomSampler(RandomSampler):
    def __init__(self, data_source, replacement=False, num_samples=None, generator=None):
        super().__init__(data_source, replacement=replacement, num_samples=num_samples, generator=generator)

    def __iter__(self):
        return iter(torch.randperm(len(self.data_source)).tolist())


def create_datasets(basepaths: list, gt_paths: list, train_eval_split: float, processor):
    datasets_train = list()
    datasets_test = list()

    for basepath, gt in zip(basepaths, gt_paths):
        df = pd.read_json(gt, lines=True)
        df.rename(columns={0: "filename", 1: "text"}, inplace=True)
        # df = df[0:2100]

        train_df, test_df = train_test_split(df, test_size=train_eval_split)
        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)

        train_dataset = HTRDataset(root_dir=basepath, df=train_df, processor=processor)
        eval_dataset = HTRDataset(root_dir=basepath, df=test_df, processor=processor)

        datasets_train.append(train_dataset)
        datasets_test.append(eval_dataset)

    len_of_datasets = sum(len(sublist) for sublist in datasets_train)

    train_concat_dataset = ConcatDataset(datasets_train)
    test_concat_dataset = ConcatDataset(datasets_test)

    return train_concat_dataset, test_concat_dataset


def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}

if __name__ == "__main__":
    
    basepaths = ["/leonardo_work/EUHPC_D02_014/data/text_recognition/HTR_1700/", "/leonardo_work/EUHPC_D02_014/data/text_recognition/police_records/", "/leonardo_work/EUHPC_D02_014/data/text_recognition/court_records/"]
    gt_paths = [os.path.join(basepath, "gt_files", "text_recognition_all_bin.jsonl") for basepath in basepaths]

    cer_metric = evaluate.load("cer")

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

    train_concat_dataset, test_concat_dataset = create_datasets(
        basepaths=basepaths, gt_paths=gt_paths, train_eval_split=0.05, processor=processor
    )
    
    print("Number of training examples:", len(train_concat_dataset))
    print("Number of validation examples:", len(test_concat_dataset))

    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")

    # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 184
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="epoch",
        num_train_epochs=3,  # change (epoch)
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,  # maybe this one
        fp16=False,
        bf16=True,
        greater_is_better=False,
        dataloader_drop_last=True,
        load_best_model_at_end=False,
        output_dir="/leonardo/home/userexternal/elenas00/projects/huggingface_htr/models/trocr_version_1",
        save_total_limit=3,
        logging_dir="/leonardo/home/userexternal/elenas00/projects/huggingface_htr/models/trocr_version_1/tensorboard",
        logging_steps=10,
        report_to="tensorboard",
        save_strategy="epoch"
    )

    train_dataloader = DataLoader(
        train_concat_dataset,
        batch_size=training_args.per_device_train_batch_size,
        sampler=CustomRandomSampler(train_concat_dataset)
    )

    val_dataloader = DataLoader(
        test_concat_dataset,
        batch_size=training_args.per_device_train_batch_size,
        sampler=CustomRandomSampler(test_concat_dataset)
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.feature_extractor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_concat_dataset,
        eval_dataset=test_concat_dataset,
        data_collator=default_data_collator,
    )
    trainer.train(resume_from_checkpoint='/leonardo/home/userexternal/elenas00/projects/huggingface_htr/models/trocr_version_1/checkpoint-9378')
