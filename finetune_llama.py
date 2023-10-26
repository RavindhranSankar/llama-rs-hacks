from dotenv import load_dotenv
import datasets
import tempfile
import logging
import random
import config
import os
import yaml
import time
import torch
import transformers
import pandas as pd
import jsonlines
from pathlib import Path
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import TrainingArguments
from transformers import AutoModelForCausalLM
from transformers import Trainer

logger = logging.getLogger(__name__)
global_config = None

# Load environment variables
env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)

# hugging face access token
ravi_hf_token = os.environ["HF_TOKEN"]
print("1: HF token read - ok")


# helper functions
def tokenize_function(data):
    text = data["user"][0] + data["output"][0]
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        padding=True,
    )

    max_length = min(tokenized_inputs["input_ids"].shape[1], 2048)
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(text, return_tensors="np", truncation=True, max_length=max_length)

    return tokenized_inputs


def inference(text, model, tokenizer, max_input_tokens=3000, max_output_tokens=2000):
    # Tokenize
    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_input_tokens)

    # Generate
    device = model.device
    generated_tokens_with_prompt = model.generate(input_ids=input_ids.to(device), max_length=max_output_tokens)

    # Decode
    generated_text_with_prompt = tokenizer.batch_decode(generated_tokens_with_prompt, skip_special_tokens=True)

    # Strip the prompt
    generated_text_answer = generated_text_with_prompt[0][len(text) :]

    return generated_text_answer


# setup
filename = "training_data_10_20_09.jsonl"
finetuning_dataset = datasets.load_dataset("json", data_files=filename, split="train")
print("2: training data read - ok")

# model, tokenizer setup
llama_2_model = "meta-llama/Llama-2-7b-chat-hf"
use_hf = True
dataset_path = "training_data_10_20_09.jsonl"
tokenizer = AutoTokenizer.from_pretrained(llama_2_model, token=ravi_hf_token)
tokenizer.pad_token = tokenizer.eos_token
max_length = 2048
print("3: tokenizer setup - okay")

# tokenize a single training data point
text = finetuning_dataset[0]["user"] + finetuning_dataset[0]["output"]
tokenized_inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True, max_length=max_length)

tokenized_dataset = finetuning_dataset.map(tokenize_function, batched=True, batch_size=1, drop_last_batch=True)
print("4: tokenized dataset done - okay")

# Setup training config
training_config = {
    "model": {"pretrained_name": llama_2_model, "max_length": 2048},
    "datasets": {"use_hf": use_hf, "path": dataset_path},
    "verbose": True,
}

# Generate train test data split
split_dataset = tokenized_dataset.train_test_split(test_size=0.1, shuffle=True, seed=123)
train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]
print("5: tokenized dataset test/train split done - okay")

# Get base_model for finetuning
base_model = AutoModelForCausalLM.from_pretrained(
    llama_2_model,
    token=ravi_hf_token,
)
print(f"6: get base model {llama_2_model} - okay")

# Setup GPU device if available
device_count = torch.cuda.device_count()
if device_count > 0:
    logger.debug("Select GPU device")
    device = torch.device("cuda")
else:
    # logger.debug("Select CPU device")
    # device = torch.device("cpu")
    raise Exception("GPU device not found !!!!")
print(f"7: get base model {llama_2_model} - okay")

base_model.to(device)
print(f"8: copy model to GPU - okay")

# Setup Training
max_steps = 100
trained_model_name = f"llama2_7b_chat_soap_notes_{max_steps}_steps"
output_dir = trained_model_name

training_args = TrainingArguments(
    # Learning rate
    learning_rate=1.0e-5,
    # Number of training epochs
    num_train_epochs=1,
    # Max steps to train for (each step is a batch of data)
    # Overrides num_train_epochs, if not -1
    max_steps=max_steps,
    # Batch size for training
    per_device_train_batch_size=1,
    # Directory to save model checkpoints
    output_dir=output_dir,
    # Other arguments
    overwrite_output_dir=False,  # Overwrite the content of the output directory
    disable_tqdm=False,  # Disable progress bars
    eval_steps=120,  # Number of update steps between two evaluations
    save_steps=120,  # After # steps model is saved
    warmup_steps=1,  # Number of warmup steps for learning rate scheduler
    per_device_eval_batch_size=1,  # Batch size for evaluation
    evaluation_strategy="steps",
    logging_strategy="steps",
    logging_steps=1,
    optim="adafactor",
    gradient_accumulation_steps=4,
    gradient_checkpointing=False,
    # Parameters for early stopping
    load_best_model_at_end=True,
    save_total_limit=1,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)
print(f"9: setup training arguments - okay")

model_flops = (
    base_model.floating_point_ops({"input_ids": torch.zeros((1, training_config["model"]["max_length"]))})
    * training_args.gradient_accumulation_steps
)
print(base_model)
print("Memory footprint", base_model.get_memory_footprint() / 1e9, "GB")
print("Flops", model_flops / 1e9, "GFLOPs")

trainer = Trainer(
    model=base_model,
    model_flops=model_flops,
    total_steps=max_steps,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)
print(f"9: setup trainer - okay")

print(f"9.5: Start training....")
training_output = trainer.train()
print(f"10: training done - okay")

save_dir = f"{output_dir}/final"
trainer.save_model(save_dir)
print("Saved model to:", save_dir)

finetuned_slightly_model = AutoModelForCausalLM.from_pretrained(save_dir, local_files_only=True)
finetuned_slightly_model.to(device)

# Run slightly finetuned model
test_question = test_dataset[0]["question"]
print("Question input (test):", test_question)

print("Finetuned slightly model's answer: ")
print(inference(test_question, finetuned_slightly_model, tokenizer))

print("---------------------------------------------")
