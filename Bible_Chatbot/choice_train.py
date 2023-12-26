import json
import argparse
import numpy as np
import csv
import random
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset, Dataset
from transformers import AutoTokenizer, BertForMultipleChoice, AutoModel, BertTokenizerFast
from tqdm import tqdm


parser = argparse.ArgumentParser(description="Finetune a transformers model on a multiple choice task")
parser.add_argument(
    "--train_file", type=str, default="./train.json", help="A csv or a json file containing the training data."
)
parser.add_argument(
    "--valid_file", type=str, default="./valid.json", help="A csv or a json file containing the validation data."
)
parser.add_argument(
    "--context_file", type=str, default="./context.json", help="A csv or a json file containing the context data."
)
parser.add_argument(
    "--model_name_or_path",
    type=str,
    default='hfl/chinese-macbert-large',#'hfl/chinese-roberta-wwm-ext', #'hfl/chinese-roberta-wwm-ext',#'shibing624/macbert4csc-base-chinese',#'uer/roberta-base-chinese-extractive-qa',#"shibing624/macbert4csc-base-chinese", #'ckiplab/bert-base-chinese-ner', #'hfl/rbt3', #'uer/roberta-base-chinese-extractive-qa', #'hfl/chinese-macbert-large', #'hfl/chinese-bert-wwm', #'hfl/chinese-roberta-wwm-ext'
    help="Path to pretrained model or model identifier from huggingface.co/models.",
    required=False,
)
parser.add_argument(
    "--tokenizer_name",
    type=str,
    default='hfl/chinese-macbert-large',#'hfl/chinese-roberta-wwm-ext', #'hfl/chinese-roberta-wwm-ext', #'hfl/chinese-macbert-large',#'shibing624/macbert4csc-base-chinese',#'uer/roberta-base-chinese-extractive-qa',#'shibing624/macbert4csc-base-chinese',#'ckiplab/bert-base-chinese-ner', #'hfl/rbt3',#'uer/roberta-base-chinese-extractive-qa',#'hfl/chinese-macbert-large', #'hfl/chinese-bert-wwm', #'hfl/chinese-roberta-wwm-ext'
    help="Pretrained tokenizer name or path if not the same as model_name",
)
parser.add_argument(
    "--rand_seed",
    type=int,
    default=2023,
    help="Random seed value setting for training",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=3e-5/2,
    help="Initial learning rate (after the potential warmup period) to use.",
)
parser.add_argument(
    "--per_device_batch_size",
    type=int,
    default=1, #3
    help="Batch size (per device) for the dataloader.",
)
parser.add_argument(
    "--model_saving_path",
    type=str,
    default='./q1',
    help="Path to save training model.",
    required=False,
)
parser.add_argument(
    "--max_seq_length",
    type=int,
    default=512, #384
    help=(
        "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
        " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
    ),
)
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=10,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
)
parser.add_argument(
    "--optimizer_gamma",
    type=float,
    default=0.999,
    help="Setting of gamma value of optimizer.",
)
parser.add_argument(
    "--num_worker",
    type=int,
    default=4,
    help="Setting of number of worker.",
)
parser.add_argument("--num_train_epochs", type=int, default=2, help="Total number of training epochs to perform.") #2
args = parser.parse_args()


tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_name, do_lower_case=True)
#tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
tokenizer.save_pretrained("./tokenizer_1/")
    

class Datasets(Dataset):
    def __init__(self, context, data, labels):
        self.data = data
        self.labels = labels
        self.context = context
        
    def __getitem__(self, index):
        label = self.labels[index]
        question = self.data[index]["question"]
        choices_id = self.data[index]["paragraphs"]
        pair = [question for i in range(len(choices_id))]
        ch_content = [self.context[i] for i in choices_id]
        
        return pair, ch_content, label
    
    def __len__(self):
        return len(self.labels)

def Data_Collator (data):
    input_ids, att_mask, token_ids = [], [], []
    for i in data:
        inputs = tokenizer(i[0], text_pair=i[1], padding="max_length", truncation=True, return_tensors="pt", max_length=args.max_seq_length)
        input_ids.append(inputs['input_ids'].tolist())
        att_mask.append(inputs['attention_mask'].tolist())
        token_ids.append(inputs['token_type_ids'].tolist())
    input_ids = torch.tensor(input_ids)
    att_mask = torch.tensor(att_mask)
    token_ids = torch.tensor(token_ids)
    labels = torch.tensor([i[2].tolist() for i in data]).long()
    return input_ids, att_mask, token_ids, labels



def Train_Q1(epoch):
    model_q1.train()
    total_train_loss = 0
    iteration = 0
    total_iteration = len(train_loader)
    for idx, (input_ids, att_mask, token_ids, label) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        
        input_ids = input_ids.to(device)
        att_mask = att_mask.to(device)
        label = label.to(device)
        out = model_q1(input_ids, attention_mask=att_mask, labels=label)
        
        total_train_loss += out.loss.item()
        out.loss.backward()
        optimizer.step()
        scheduler.step()

        iteration += 1
        if(iteration % 500 ==0):
            print("Epoch: {}, Iteration number: {}, Loss: {:.5f}, Progress: {:.3f}".format(epoch, iteration, out.loss.item(), iteration/total_iteration*100))
        
    print("Epoch: {}, Average Training Loss: {:.5f}".format(epoch, total_train_loss/len(train_loader)))

def Valid_Q1():
    model_q1.eval()
    total_eval_acc = 0
    total_eval_loss = 0
    for idx, (input_ids, att_mask, _, label) in enumerate(tqdm(valid_loader)):
        with torch.no_grad():
            input_ids = input_ids.to(device)
            att_mask = att_mask.to(device)
            label = label.to(device)
            out = model_q1(input_ids, attention_mask=att_mask, labels=label)
        loss = out.loss

        total_eval_loss += loss.item()
        total_eval_acc += (out[1].argmax(1).data == label.data).float().mean().item()
        
    avg_valid_acc = total_eval_acc / len(valid_loader)
    print("Average Validation Accuracy: {:.5f}".format(avg_valid_acc))
    print("Average Validation Loss: {:.5f}".format(total_eval_loss/len(valid_loader)))
    print("--------------------------------------------")
    
def Generate_Labels(json_file):
    
    labels = torch.tensor([])
    for i in json_file:
        gt_label_idx = i["paragraphs"].index(i["relevant"])
        labels = torch.cat((labels, torch.tensor(gt_label_idx).unsqueeze(0)), 0)
    
    return labels
    

if __name__ == '__main__':
    
    random_seed = args.rand_seed
    
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    torch.manual_seed(random_seed)
    
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    
    #Set whether torch can use GPU
    if torch.cuda.is_available():
        device = 'cuda'
    else: 
        device = 'cpu'
    
    #device = 'cpu'
    print("Training Device: " + device)
	
	
    with open(args.context_file, newline='') as context_file:
        context = json.load(context_file)
    with open(args.train_file, newline='') as train_file:
        train_json_file = json.load(train_file)
    with open(args.valid_file, newline='') as valid_file:
        valid_json_file = json.load(valid_file)
    
    
    train_labels = Generate_Labels(train_json_file)
    valid_labels = Generate_Labels(valid_json_file)
    
    
    train_dataset = Datasets(context, train_json_file, train_labels)
    valid_dataset = Datasets(context, valid_json_file, valid_labels)
    train_loader = DataLoader(dataset=train_dataset , batch_size= args.per_device_batch_size, shuffle=True, num_workers=args.num_worker, collate_fn=Data_Collator)
    valid_loader = DataLoader(dataset=valid_dataset , batch_size= args.per_device_batch_size, shuffle=True, num_workers=args.num_worker, collate_fn=Data_Collator)

    model_q1 = BertForMultipleChoice.from_pretrained(args.model_name_or_path)
    #model_q1 = AutoModel.from_pretrained(args.model_name_or_path)
    
    model_q1.to(device)
    optimizer = optim.AdamW(model_q1.parameters(), lr=args.learning_rate)
    scheduler = StepLR(optimizer, step_size=args.gradient_accumulation_steps, gamma=args.optimizer_gamma)
    
    
    for e in range(args.num_train_epochs):
        print("---------------- Epoch: {} ----------------".format(e))
        Train_Q1(e)
    
    Valid_Q1()
    
    model_saving_dir = args.model_saving_path
    model_q1.save_pretrained(model_saving_dir)
    