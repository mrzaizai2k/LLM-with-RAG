
import sys
sys.path.append("")

from datasets import load_dataset, ClassLabel
from setfit import SetFitModel, Trainer, TrainingArguments
from src.utils import config_parser

config = config_parser(data_config_path = 'config/gpt_routing_train_config.yaml')
dataset_name = config['dataset_name']
sentence_model = config['sentence_model']
text_col_name = config['text_col_name']
label_col_name = config['label_col_name']

# validation_split = 25
datasets = load_dataset(dataset_name, split="train", cache_dir=True)

# Assuming the loaded dataset is named "datasets"
new_features = datasets.features.copy()  # Get dataset features
new_features[label_col_name] = ClassLabel(num_classes = len(datasets.unique(label_col_name)), names=datasets.unique(label_col_name))  # Extract unique labels and cast
datasets = datasets.cast(new_features)
ddict  = datasets.train_test_split(test_size=config['test_size'], stratify_by_column=label_col_name, shuffle =True)
train_ds, val_ds = ddict["train"], ddict["test"]
ddict_val_test  = val_ds.train_test_split(test_size=0.5, stratify_by_column=label_col_name, shuffle =True)
val_ds, test_ds = ddict_val_test["train"], ddict_val_test["test"]

# Load a SetFit model from Hub
model = SetFitModel.from_pretrained(
    sentence_model,
    labels=datasets.unique(label_col_name),
)

args = TrainingArguments(
    batch_size=config['batch_size'],
    num_epochs=config['num_epochs'],
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

args.eval_strategy = args.evaluation_strategy # SetFitpackage error

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    metric="accuracy",
    column_mapping={"text": text_col_name, "label": label_col_name}  # Map dataset columns to text/label expected by trainer
)

trainer.train()

metrics = trainer.evaluate(test_ds)
print("metrics", metrics)

if config['push_to_hub']:
    trainer.push_to_hub(config['huggingface_out_dir'])

preds=model.predict(["who has the pen","The birthday of Kerger", "explain in detail the karger min cut and it's complexity"])
print("Prediction:", preds)




