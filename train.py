from datasets import  DatasetDict, Dataset,  enable_progress_bar
import random
import time
import sys
import utils as ut
from transformers import WhisperProcessor as WP
from transformers import WhisperForConditionalGeneration as WCG
import torch
from dataclasses import dataclass
from typing import Any,Dict,List,Union
import config as cf
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

def main():
    ### config of the model
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    processor = WP.from_pretrained("openai/whisper-large-v3-turbo", language="English", task="transcribe")
    model = WCG.from_pretrained("openai/whisper-large-v3-turbo")
    model.generation_config.language = "english"
    model.generation_config.task = 'transcribe'
    model.generation_config.forced_decoder_ids = None




    ### script to get the random array of examples amounting up to a given time
    random.seed(time.time())
    enable_progress_bar()

   
    
    sys.stdout.write(f"\rProgress: Importing data...")
    sys.stdout.flush()
    stream = ut.build_stream()
    #  not needed after changing the dataset_download
    # sys.stdout.write(f"\rProgress: Resampling audio...")
    # sys.stdout.flush()
    # stream  = stream.cast_column("audio", Audio(sampling_rate=16000))
   

    sys.stdout.write(f"\rProgress: Sampling data...")
    sys.stdout.flush()
    stream = ut.get_files(stream)

    sys.stdout.write(f"\rProgress: Writing the Datasets...")
    sys.stdout.flush()
    data = DatasetDict({
        "train": Dataset.from_list(stream["train"]),  
        "test":  Dataset.from_list(stream["test"])    
    })
    print("Stream after import:\n")
    print(data)
    
    
    sys.stdout.write(f"\rProgress: Deleting columns...")
    sys.stdout.flush()
    data = data.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])
    print("Stream after deletion of columns:\n")
    print(data)
    ### preparing data for the model
  
    
    # ### ensuring that the format of the audio is correct to make the calculations more optimal
    # for split in data.keys():
    #     data[split].set_format(type="numpy", columns=["audio"])

    sys.stdout.write(f"Progress: Mapping the dataset...")
    sys.stdout.flush()
    data = data.map(ut.prepare_dataset, remove_columns=data["train"].column_names , num_proc=7, desc = "Mapping")
    ###loading the model
    ### custom data class
    
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # example performance for an rtx 3090 with r7 5700X @75w
    # per_device_train_batch_size=16,
    # gradient_accumulation_steps=1, 26,28s/it 33GB VRAM
    # per_device_eval_batch_size=6,
    # gradient_checkpointing=False, 


    # per_device_train_batch_size=16,
    # gradient_accumulation_steps=1,  7s/it 20,8GB VRAM
    # gradient_checkpointing=True,
    # per_device_eval_batch_size=8,

    # per_device_train_batch_size=18,
    # gradient_accumulation_steps=1,  7,8s/it 21GB VRAM 
    # gradient_checkpointing=True,
    # per_device_eval_batch_size=8,

    training_args = Seq2SeqTrainingArguments(
        output_dir=cf.get_model_path(),  # change to a repo name of your choice
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        gradient_checkpointing=True,
        per_device_eval_batch_size=8,
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=1500,
        bf16=True,
        eval_strategy="steps",
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=500,#saving steps have to be a round multiple of the evaluation steps, 
        eval_steps=500,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        # dataloader_num_workers=6,  # Use 6 CPU cores for data loading - crashes on windows
        dataloader_pin_memory=True,
        # torch_compile=True, #crashes on windows

    )
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=data["train"],
        eval_dataset=data["test"],
        data_collator=data_collator,
        compute_metrics=ut.compute_metrics,
        processing_class=processor,
    )

    trainer.train()







### running the model
# sys.stdout.write(f"\rProgress: Encoding shit...")
# sys.stdout.flush()
# input_str = data["train"][0]["sentence"]
# labels = tokenizer(input_str).input_ids
# decoded_with_special = tokenizer.decode(labels,skip_special_tokens = False)
# decoded_str= tokenizer.decode(labels,skip_special_tokens = True)

# print(f"Input:                 {input_str}")
# print(f"Decoded w/ special:    {decoded_with_special}")
# print(f"Decoded w/out special: {decoded_str}")


if __name__ == "__main__":
    main()

    


