from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.utils.data
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm, tqdm_notebook
import os
import warnings
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertAdam
from pytorch_pretrained_bert import BertConfig
device = torch.device('cuda')


def convert_lines(example, max_seq_length,tokenizer):
    max_seq_length -=2
    all_tokens = []
    longer = 0
    for text in tqdm(example):
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a)>max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
        all_tokens.append(one_token)
    return np.array(all_tokens)


MAX_SEQUENCE_LENGTH = 220
SEED = 1234
BATCH_SIZE = 32
BERT_MODEL_PATH = './datas/uncased_L-12_H-768_A-12'

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

EPOCHS = 1
TOXICITY_COLUMN = 'target'
num_to_load = 1000000
valid_size = 100000

bert_config = BertConfig('./datas/bert_config.json')
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, do_lower_case=True)


train_df = pd.read_csv("./datas/train.csv")
print('train %d records' % len(train_df))
train_df['comment_text'] = train_df['comment_text'].astype(str)

valid_df = pd.read_csv("./datas/train_multi.csv")
valid_df['comment_text'] = valid_df['comment_text'].astype(str)
print('valid %d records' % len(valid_df))

train_seqs = convert_lines(train_df["comment_text"].fillna("DUMMY_VALUE"), MAX_SEQUENCE_LENGTH, tokenizer)
valid_seqs = convert_lines(valid_df["comment_text"].fillna("DUMMY_VALUE"), MAX_SEQUENCE_LENGTH, tokenizer)
train_df=train_df.fillna(0)
identity_columns = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
y_columns=['target']

train_df = train_df.drop(['comment_text'],axis=1)
train_df['target']=(train_df['target']>=0.5).astype(float)

valid_df = valid_df.fillna(0)
valid_df = valid_df.drop(['comment_text'],axis=1)
valid_df['target'] = (valid_df['toxic'] == 1) | (valid_df['severe_toxic'] == 1)
valid_df['target'] = valid_df['target'] | (valid_df['obscene'] == 1)
valid_df['target'] = valid_df['target'] | (valid_df['threat'] == 1)
valid_df['target'] = valid_df['target'] | (valid_df['insult'] == 1)
valid_df['target'] = valid_df['target'] | (valid_df['identity_hate'] == 1)
valid_df['target'] = valid_df['target'].astype(float)

model = BertForSequenceClassification(bert_config, num_labels=1)
model.load_state_dict(torch.load("./datas/bert_pytorch.bin"))
model.to(device)
for param in model.parameters():
    param.requires_grad = False

X = train_seqs
y = train_df['target']
train_dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.float))

output_model_file = "./datas/mybert.bin"
lr = 2e-5
batch_size = 32
accumulation_steps = 2
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

lr_layers = [1 for i in range(12)]
for i in range(len(lr_layers)):
    lr_layers[i] = lr_layers[i - 1] * 0.95 if i > 0 else lr
lr_layers = lr_layers[::-1]

model.zero_grad()
model = model.to(device)
param_optimizer = list(model.named_parameters())

import re
pattern = re.compile('[0-9]+')

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if n.find(".0.") != -1 and not any(nd in n for nd in no_decay)], 'lr': lr_layers[0], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if n.find(".0.") != -1 and any(nd in n for nd in no_decay)], 'lr': lr_layers[0], 'weight_decay': 0.0},
    {'params': [p for n, p in param_optimizer if n.find(".1.") != -1 and not any(nd in n for nd in no_decay)], 'lr': lr_layers[1], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if n.find(".1.") != -1 and any(nd in n for nd in no_decay)], 'lr': lr_layers[1], 'weight_decay': 0.0},
    {'params': [p for n, p in param_optimizer if n.find(".2.") != -1 and not any(nd in n for nd in no_decay)], 'lr': lr_layers[2], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if n.find(".2.") != -1 and any(nd in n for nd in no_decay)], 'lr': lr_layers[2], 'weight_decay': 0.0},
    {'params': [p for n, p in param_optimizer if n.find(".3.") != -1 and not any(nd in n for nd in no_decay)], 'lr': lr_layers[3], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if n.find(".3.") != -1 and any(nd in n for nd in no_decay)], 'lr': lr_layers[3], 'weight_decay': 0.0},
    {'params': [p for n, p in param_optimizer if n.find(".4.") != -1 and not any(nd in n for nd in no_decay)], 'lr': lr_layers[4], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if n.find(".4.") != -1 and any(nd in n for nd in no_decay)], 'lr': lr_layers[4], 'weight_decay': 0.0},
    {'params': [p for n, p in param_optimizer if n.find(".5.") != -1 and not any(nd in n for nd in no_decay)], 'lr': lr_layers[5], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if n.find(".5.") != -1 and any(nd in n for nd in no_decay)], 'lr': lr_layers[5], 'weight_decay': 0.0},
    {'params': [p for n, p in param_optimizer if n.find(".6.") != -1 and not any(nd in n for nd in no_decay)], 'lr': lr_layers[6], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if n.find(".6.") != -1 and any(nd in n for nd in no_decay)], 'lr': lr_layers[6], 'weight_decay': 0.0},
    {'params': [p for n, p in param_optimizer if n.find(".7.") != -1 and not any(nd in n for nd in no_decay)], 'lr': lr_layers[7], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if n.find(".7.") != -1 and any(nd in n for nd in no_decay)], 'lr': lr_layers[7], 'weight_decay': 0.0},
    {'params': [p for n, p in param_optimizer if n.find(".8.") != -1 and not any(nd in n for nd in no_decay)], 'lr': lr_layers[8], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if n.find(".8.") != -1 and any(nd in n for nd in no_decay)], 'lr': lr_layers[8], 'weight_decay': 0.0},
    {'params': [p for n, p in param_optimizer if n.find(".9.") != -1 and not any(nd in n for nd in no_decay)], 'lr': lr_layers[9], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if n.find(".9.") != -1 and any(nd in n for nd in no_decay)], 'lr': lr_layers[9], 'weight_decay': 0.0},
    {'params': [p for n, p in param_optimizer if n.find(".10.") != -1 and not any(nd in n for nd in no_decay)], 'lr': lr_layers[10], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if n.find(".10.") != -1 and any(nd in n for nd in no_decay)], 'lr': lr_layers[10], 'weight_decay': 0.0},
    {'params': [p for n, p in param_optimizer if n.find(".11.") != -1 and not any(nd in n for nd in no_decay)], 'lr': lr_layers[11], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if n.find(".11.") != -1 and any(nd in n for nd in no_decay)], 'lr': lr_layers[11], 'weight_decay': 0.0},
    {'params': [p for n, p in param_optimizer if not pattern.findall(n) and not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if not pattern.findall(n) and any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
train = train_dataset
num_train_optimization_steps = int(EPOCHS*len(train)/batch_size/accumulation_steps)
optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=lr,
                     warmup=0.05,
                     t_total=num_train_optimization_steps)
model=model.train()

tq = tqdm_notebook(range(EPOCHS))
for epoch in tq:
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    avg_loss = 0.
    avg_accuracy = 0.
    lossf = None
    tk0 = tqdm_notebook(enumerate(train_loader),total=len(train_loader),leave=False)
    optimizer.zero_grad()   # Bug fix - thanks to @chinhuic
    for i,(x_batch, y_batch) in tk0:
#        optimizer.zero_grad()
        y_pred = model(x_batch.to(device), attention_mask=(x_batch>0).to(device), labels=None)
        loss = F.binary_cross_entropy_with_logits(y_pred,y_batch.to(device))
        if (i+1) % accumulation_steps == 0:             # Wait for several backward steps
            optimizer.step()                            # Now we can do an optimizer step
            optimizer.zero_grad()
        if lossf:
            lossf = 0.98*lossf+0.02*loss.item()
        else:
            lossf = loss.item()
        tk0.set_postfix(loss = lossf)
        avg_loss += loss.item() / len(train_loader)
        avg_accuracy += torch.mean(((torch.sigmoid(y_pred[:,0])>0.5) == (y_batch[:,0]>0.5).to(device)).to(torch.float) ).item()/len(train_loader)
    tq.set_postfix(avg_loss=avg_loss,avg_accuracy=avg_accuracy)


torch.save(model.state_dict(), output_model_file)