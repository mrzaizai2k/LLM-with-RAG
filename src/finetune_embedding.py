import sys
sys.path.append("")

# model
from sentence_transformers import SentenceTransformer, models, InputExample, losses
from datasets import load_dataset
from torch.utils.data import DataLoader
from Utils.utils import *

device = take_device()
config = config_parser(data_config_path = 'config/finetune_embedding.yaml')
save_path = config['save_path']
model_type = config['model_type']

model = SentenceTransformer(model_type)
model.to(device)

word_embedding_model = models.Transformer(model_type, max_seq_length=config['max_seq_length'])
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


dataset = load_dataset(config['dataset_id'])

train_examples = []
train_data = dataset['train']['set']

for i in range(0, len(train_data)):
    example = train_data[i]
    train_examples.append(InputExample(texts=[example['query'],example['pos'][0],example['neg'][0]]))


train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=config['batch_size'])
train_loss = losses.TripletLoss(model)


model.fit(train_objectives=[(train_dataloader, train_loss)], 
          epochs=4,
          output_path = save_path)

print(f"Done fine tuning embedding model. The modle saved in {save_path}")