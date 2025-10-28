import os
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

from datasets import load_dataset
# import numpy as np

from transformers import AutoTokenizer, TrainingArguments, AutoModelForSequenceClassification, Trainer

from transformers import TrainerCallback, logging


# import logging
# import transformers

from pprint import pprint

# from datasets import load_metric
import evaluate

from torchinfo import summary

# -----------------------------------------------------------------------------

# Get and print the current working directory
current_working_directory = os.getcwd()
print(f"The current working directory is: {current_working_directory}")

# Change the working directory to a new directory (replace with the path you want)
new_working_directory = "C:/1-eigenes/Lehrauftrag - BHT/Vorlesungsreihe WS25-26/WS2526_Lec7_ Sentiment Analyzer _BERT/0_Exercise_env"
os.chdir(new_working_directory)

# Get and print the new current working directory
new_current_working_directory = os.getcwd()
print(f"The new current working directory is: {new_current_working_directory}")

# -----------------------------------------------------------------------------

class VerboseCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            print(f"Step {state.global_step}: {logs}")


def tokenize_fn(batch):
  return tokenizer(batch['sentence'], truncation=True)

def compute_metrics(logits_and_labels):
  # metric = load_metric("glue", "sst2")
  logits, labels = logits_and_labels
  predictions = np.argmax(logits, axis=-1)
  return metric.compute(predictions=predictions, references=labels)

# -----------------------------------------------------------------------------


raw_datasets = load_dataset("glue", "sst2")

raw_datasets
raw_datasets['train']
dir(raw_datasets['train'])
type(raw_datasets['train'])


raw_datasets['train'].data
raw_datasets['train'][0]
raw_datasets['train'][50000:50003]

raw_datasets['train'].features

# -----------------------------------------------------------------------------

# checkpoint = "bert-base-uncased"
checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

tokenized_sentences = tokenizer(raw_datasets['train'][0:3]['sentence'])

pprint(tokenized_sentences)


tokenized_datasets = raw_datasets.map(tokenize_fn, batched=True)







# training_args = TrainingArguments(
#   output_dir='my_trainer',
#   eval_strategy='epoch',
#   save_strategy='epoch',
#   num_train_epochs=1,
#   per_device_train_batch_size=8,  # Adjust based on your GPU memory
#   per_device_eval_batch_size=8,
#   logging_dir='./logs',
#   logging_steps=500,
#   load_best_model_at_end=True,
# )

# training_args = TrainingArguments(
#   'my_trainer',
#   evaluation_strategy='epoch',
#   save_strategy='epoch',
#   num_train_epochs=1,
# )


model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=2)




type(model)
model

summary(model)


# Set verbose logging
logging.set_verbosity_info()

training_args = TrainingArguments(
  output_dir='my_trainer',
  eval_strategy='epoch',
  save_strategy='epoch',
  num_train_epochs=1,
  logging_steps=50,  # Adjust based on dataset size
  logging_first_step=True,
  logging_dir='./logs',
  report_to='none',  # or 'tensorboard' if you want to use it
  disable_tqdm=False,  # Keep progress bars enabled
)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


# -----------------------------------------------------------------------------

params_before = []
for name, p in model.named_parameters():
  params_before.append(p.detach().cpu().numpy())



# metric = load_metric("glue", "sst2")
# metric.compute(predictions=[1, 0, 1], references=[1, 0, 0])


# Changed from load_metric to evaluate.load
metric = evaluate.load("glue", "sst2")
metric.compute(predictions=[1, 0, 1], references=[1, 0, 0])


# -----------------------------------------------------------------------------
# Train and save BERT model

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()


trainer.save_model('my_saved_model3')


