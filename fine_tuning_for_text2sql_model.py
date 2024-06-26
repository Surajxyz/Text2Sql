from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, load_dataset, interleave_datasets, load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
import warnings
warnings.filterwarnings("ignore")

torch.cuda.is_available()

model_name='t5-small'

tokenizer = AutoTokenizer.from_pretrained(model_name)

original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
original_model = original_model

dataset_scc_train = load_dataset("b-mc2/sql-create-context", split='train[:80%]')
dataset_scc_test  = load_dataset("b-mc2/sql-create-context", split='train[-20%:-10%]')
dataset_scc_val   = load_dataset("b-mc2/sql-create-context", split='train[-10%:]')

dataset_tts_train = load_dataset("Clinton/Text-to-sql-v1", split='train[:80%]')
dataset_tts_train = dataset_tts_train.remove_columns(['source', 'text'])
dataset_tts_train = dataset_tts_train.rename_columns({'instruction': 'question', 'input': 'context', 'response': 'answer'})
dataset_tts_test  = load_dataset("Clinton/Text-to-sql-v1", split='train[-20%:-10%]')
dataset_tts_test  = dataset_tts_test.remove_columns(['source', 'text'])
dataset_tts_test  = dataset_tts_test.rename_columns({'instruction': 'question', 'input': 'context', 'response': 'answer'})
dataset_tts_val   = load_dataset("Clinton/Text-to-sql-v1", split='train[-10%:]')
dataset_tts_val   = dataset_tts_val.remove_columns(['source', 'text'])
dataset_tts_val   = dataset_tts_val.rename_columns({'instruction': 'question', 'input': 'context', 'response': 'answer'})

dataset_ks_train  = load_dataset("knowrohit07/know_sql", split='validation[:80%]')
dataset_ks_test   = load_dataset("knowrohit07/know_sql", split='validation[-20%:-10%]')
dataset_ks_val    = load_dataset("knowrohit07/know_sql", split='validation[-10%:]')

dataset = DatasetDict({ 'train': interleave_datasets([dataset_scc_train, dataset_tts_train, dataset_ks_train]),
                        'test': interleave_datasets([dataset_scc_test, dataset_tts_test, dataset_ks_test]),
                        'validation': interleave_datasets([dataset_scc_val, dataset_tts_val, dataset_ks_val])})

dataset

dataset['test'][0]

def tokenize_function(example):
    start_prompt = "Tables:\n"
    middle_prompt = "\n\nQuestion:\n"
    end_prompt = "\n\nAnswer:\n"

    data_zip = zip(example['context'], example['question'])
    prompt = [start_prompt + context + middle_prompt + question + end_prompt for context, question in data_zip]
    example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
    example['labels'] = tokenizer(example['answer'], padding="max_length", truncation=True, return_tensors="pt").input_ids

    return example

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['question', 'context', 'answer'])

finetuned_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
finetuned_model = finetuned_model
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenized_datasets['test']

output_dir="/content/Result"
training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=5e-3,
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    logging_steps=50,
    evaluation_strategy='steps',
    eval_steps=500,
)

trainer = Trainer(
    model=finetuned_model,
    args=training_args,
    train_dataset=tokenized_datasets['test'],
    # here test dataset is used because compute is note there so we are not using train dataset (tokenized_datasets['test'])
    eval_dataset=tokenized_datasets['validation'],
)

trainer.train()

finetuned_model.save_pretrained("finetuned_model_1_epoch")

