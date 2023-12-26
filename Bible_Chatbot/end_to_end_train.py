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
from transformers import AutoTokenizer, BertForMultipleChoice, BertForQuestionAnswering, BertConfig, AutoModel, BertTokenizerFast
from transformers import CONFIG_MAPPING, AutoTokenizer, BertForQuestionAnswering, BertConfig, AutoModelForQuestionAnswering, BertTokenizerFast, BertTokenizer, AutoConfig
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

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
    "--model_name_or_path_1",
    type=str,
    default='hfl/chinese-macbert-large',#'deepset/roberta-base-squad2',#'hfl/chinese-macbert-large', #'hfl/chinese-bert-wwm', #'hfl/chinese-roberta-wwm-ext'
    help="Path to pretrained model or model identifier from huggingface.co/models.",
    required=False,
)
parser.add_argument(
    "--model_name_or_path_2",
    type=str,
    default='hfl/chinese-macbert-large',#'deepset/roberta-base-squad2',#'hfl/chinese-macbert-large', #'hfl/chinese-bert-wwm', #'hfl/chinese-roberta-wwm-ext'
    help="Path to pretrained model or model identifier from huggingface.co/models.",
    required=False,
)
parser.add_argument(
    "--tokenizer_name_1",
    type=str,
    default='hfl/chinese-macbert-large',#'hfl/chinese-macbert-large', #'hfl/chinese-bert-wwm', #'hfl/chinese-roberta-wwm-ext'
    help="Pretrained tokenizer name or path if not the same as model_name",
)
parser.add_argument(
    "--tokenizer_name_2",
    type=str,
    default='hfl/chinese-macbert-large',#'hfl/chinese-macbert-large', #'hfl/chinese-bert-wwm', #'hfl/chinese-roberta-wwm-ext'
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
    "--per_device_batch_size_1",
    type=int,
    default=1,
    help="Batch size (per device) for the dataloader.",
)
parser.add_argument(
    "--per_device_batch_size_2",
    type=int,
    default=4,
    help="Batch size (per device) for the dataloader.",
)

parser.add_argument(
    "--model_saving_path_1",
    type=str,
    default='./all/q1',
    help="Path to save training model q1.",
    required=False,
)
parser.add_argument(
    "--model_saving_path_2",
    type=str,
    default='./all/q2',
    help="Path to save training model q2.",
    required=False,
)
parser.add_argument(
    "--max_seq_length",
    type=int,
    default=445, #384
    help=(
        "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
        " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
    ),
)
parser.add_argument(
    "--max_q_seq_length",
    type=int,
    default=64, #64
    help=(
        "The maximum total input sequence length in question after tokenization. Sequences longer than this will be truncated,"
        " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
    ),
)
parser.add_argument(
    "--max_p_seq_length",
    type=int,
    default=445, #384
    help=(
        "The maximum total input sequence length in paragraph after tokenization. Sequences longer than this will be truncated,"
        " sequences shorter will be padded if `--pad0_to_max_lengh` is passed."
    ),
)
parser.add_argument(
    "--doc_stride",
    type=int,
    default=128, #128
    help=(
        "The maximum input stride for model to process document."
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
parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs to perform.") #3
parser.add_argument("--num_fine_epochs", type=int, default=2, help="Total number of training epochs to perform.") #3

args = parser.parse_args()

def Save_model():
    model_save_dir_1 = args.model_saving_path_1
    model_q1.save_pretrained(model_save_dir_1)
    
    model_save_dir_2 = args.model_saving_path_2
    model_q2.save_pretrained(model_save_dir_2)


class Datasets_Q1(Dataset):
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

def Data_Collator(data):
    input_ids, att_mask, token_ids = [], [], []
    for i in data:
        inputs = tokenizer_1(i[0], text_pair=i[1], padding="max_length", truncation=True, return_tensors="pt", max_length=args.max_seq_length)
        input_ids.append(inputs['input_ids'].tolist())
        att_mask.append(inputs['attention_mask'].tolist())
        token_ids.append(inputs['token_type_ids'].tolist())
    input_ids = torch.tensor(input_ids)
    att_mask = torch.tensor(att_mask)
    token_ids = torch.tensor(token_ids)
    labels = torch.tensor([i[2].tolist() for i in data]).long()
    return input_ids, att_mask, token_ids, labels

def Generate_Labels(json_file):
    
    labels = torch.tensor([])
    for i in json_file:
        gt_label_idx = i["paragraphs"].index(i["relevant"])
        labels = torch.cat((labels, torch.tensor(gt_label_idx).unsqueeze(0)), 0)
    
    return labels

class Dataset_Q2(Dataset):

    def __init__(self, questions, token_q, token_p):
        self.questions = questions
        self.token_q = token_q
        self.token_p = token_p
        self.max_q_len = args.max_q_seq_length
        self.max_p_len = args.max_p_seq_length
        self.doc_stride =  args.doc_stride
        # Input sequence length = [CLS] + question + [SEP] + paragraph + [SEP]
        self.max_seq_len = 1 + self.max_q_len + 1 + self.max_p_len + 1

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        token_q = self.token_q[idx]
        token_p = self.token_p[question["relevant"]]

        input_ids_lst, token_type_ids_lst, att_mask_lst = [], [], []
        
        # Paragraph is split into several windows, each with start positions separated by step "doc_stride"
        for i in range(0, len(token_p), self.doc_stride):
            
            # Slice question/paragraph and add special tokens (101: CLS, 102: SEP)
            input_ids_q = [101] + token_q.ids[:self.max_q_len] + [102] 
            #For long questions, take the last part and count it back as valid text.
            #input_ids_q = [101] + token_q.ids[-self.max_q_len:] + [102]
            input_ids_p = token_p.ids[i : i + self.max_p_len] + [102]
            
            # Pad sequence and obtain inputs to model
            input_ids, token_type_ids, att_mask = self.padding(input_ids_q, input_ids_p)
            
            input_ids_lst.append(input_ids)
            token_type_ids_lst.append(token_type_ids)
            att_mask_lst.append(att_mask)
        
        return torch.tensor(input_ids_lst), torch.tensor(token_type_ids_lst), torch.tensor(att_mask_lst)

    def padding(self, input_ids_q, input_ids_p):
        # Pad zeros if sequence length is shorter than max_seq_len
        pad_len = self.max_seq_len - len(input_ids_q) - len(input_ids_p)
        # Indices of input sequence tokens in the vocabulary
        input_ids = input_ids_q + input_ids_p + [0] * pad_len
        # Segment token indices to indicate first and second portions of the inputs. Indices are selected in [0, 1]
        token_type_ids = [0] * len(input_ids_q) + [1] * len(input_ids_p) + [0] * pad_len
        # Mask to avoid performing attention on padding token indices. Mask values selected in [0, 1]
        att_mask = [1] * (len(input_ids_q) + len(input_ids_p)) + [0] * pad_len
        
        return input_ids, token_type_ids, att_mask


class Datasets_Q2(Dataset):

    def __init__(self, split_idx, questions, token_q, token_p):
        self.split_idx = split_idx
        self.questions = questions
        self.token_q = token_q
        self.token_p = token_p
        self.max_q_len = args.max_q_seq_length
        self.max_p_len = args.max_p_seq_length
        self.doc_stride = args.doc_stride
        self.max_seq_len = 1 + self.max_q_len + 1 + self.max_p_len + 1

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        q = self.questions[idx]
        token_q = self.token_q[idx]
        token_p = self.token_p[q["relevant"]]

        if self.split_idx == "valid" or self.split_idx == "train":
            # Convert answer's start/end positions in paragraph_text to start/end positions in tokenized_paragraph  
            ans_start_token = token_p.char_to_token(q["answer"]["start"])
            ans_end_token = token_p.char_to_token(q["answer"]["start"] + len(q["answer"]["text"]) -1)

            # A single window is obtained by slicing the portion of paragraph containing the answer
            middle = (ans_start_token + ans_end_token) // 2
            p_start = max(0, min(middle - self.max_p_len // 2, len(token_p) - self.max_p_len))
            p_end = p_start + self.max_p_len
            
            # Slice question/paragraph and add special tokens (101: CLS, 102: SEP)
            input_ids_q = [101] + token_q.ids[:self.max_q_len] + [102] 
            #For long questions, take the last part and count it back as valid text.
            #input_ids_q = [101] + token_q.ids[-self.max_q_len:] + [102] 
            input_ids_p = token_p.ids[p_start : p_end] + [102]		
            
            # Convert answer's start/end positions in tokenized_paragraph to start/end positions in the window  
            ans_start_token += len(input_ids_q) - p_start
            ans_end_token += len(input_ids_q) - p_start
            
            # Pad sequence and obtain inputs to model 
            input_ids, token_type_ids, att_mask = self.padding(input_ids_q, input_ids_p)
            return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(att_mask), ans_start_token, ans_end_token

        else:
            input_ids_lst, token_type_ids_lst, att_mask_lst = [], [], []
            # Paragraph is split into several windows, each with start positions separated by step "doc_stride"
            for i in range(0, len(token_p), self.doc_stride):
                # Slice question/paragraph and add special tokens (101: CLS, 102: SEP)
                input_ids_q = [101] + token_q.ids[:self.max_q_len] + [102] 
                #For long questions, take the last part and count it back as valid text.
                #input_ids_question = [101] + tokenized_question.ids[-self.max_question_len:] + [102]
                input_ids_p = token_p.ids[i : i + self.max_p_len] + [102]
                # Pad sequence and obtain inputs to model
                input_ids, token_type_ids, att_mask = self.padding(input_ids_q, input_ids_p)
                
                input_ids_lst.append(input_ids)
                token_type_ids_lst.append(token_type_ids)
                att_mask_lst.append(att_mask)
            
            return torch.tensor(input_ids_lst), torch.tensor(token_type_ids_lst), torch.tensor(att_mask_lst), idx
        
        

    def padding(self, input_ids_q, input_ids_p):
        # Pad zeros if sequence length is shorter than max_seq_len
        pad_len = self.max_seq_len - len(input_ids_q) - len(input_ids_p)
        # Indices of input sequence tokens in the vocabulary
        input_ids = input_ids_q + input_ids_p + [0] * pad_len
        # Segment token indices to indicate first and second portions of the inputs. Indices are selected in [0, 1]
        token_type_ids = [0] * len(input_ids_q) + [1] * len(input_ids_p) + [0] * pad_len
        # Mask to avoid performing attention on padding token indices. Mask values selected in [0, 1]
        att_mask = [1] * (len(input_ids_q) + len(input_ids_p)) + [0] * pad_len
        
        return input_ids, token_type_ids, att_mask

    def generate_token(json_file):
        return tokenizer_1([line["question"] for line in json_file], add_special_tokens=False)
    def generate_context_token(context_file):
        return tokenizer_1(context_file, add_special_tokens=False)

def evaluation_q2(data, output):
    ans = ''
    max_prob = float('-inf')
    num_of_windows = data[0].shape[1]
        
    for k in range(num_of_windows):
        start_prob, start_index = torch.max(output.start_logits[k], dim=0)
        end_prob, end_index = torch.max(output.end_logits[k], dim=0)
        
        prob = start_prob + end_prob
        if start_index > end_index:
            continue
        if prob > max_prob:
            max_prob = prob
            ans = tokenizer_1.decode(data[0][0][k][start_index:end_index + 1])
    
    return ans.replace(' ','')

def Train_ALL():
    step_0 = 0
    step_1 = 0
    train_loss = train_acc  =0
    
    model_q2.train()
    model_q1.train()
    
    
    sel_train_loss = 0
    total_iteration = len(train_loader_q1)
    
    for idx, (input_ids, att_mask, token_ids, label) in enumerate(tqdm(train_loader_q1)):
        optimizer_1.zero_grad()
            
        input_ids = input_ids.to(device_0)
        att_mask = att_mask.to(device_0)
        label = label.to(device_0)
        out = model_q1(input_ids, attention_mask=att_mask, labels=label)
            
        sel_train_loss += out.loss.item()
        out.loss.backward()
        optimizer_1.step()
        scheduler_1.step()

        step_0 += 1
        if(step_0 % 500 ==0):
            print("Epoch: {}, Iteration number: {}, Loss: {:.5f}, Progress: {:.3f}".format(e, step_0, out.loss.item(), step_0/total_iteration*100))
    print("Epoch: {}, Average Selection Training Loss: {:.5f}".format(e, sel_train_loss/len(train_loader_q1))) 
        
    
    for data in tqdm(train_loader):
        
        optimizer_2.zero_grad()
        
        data = [i.to(device_1) for i in data]
        output = model_q2(input_ids=data[0], token_type_ids=data[1], attention_mask=data[2], start_positions=data[3], end_positions=data[4])
        start_idx = torch.argmax(output.start_logits, dim=1)
        end_idx = torch.argmax(output.end_logits, dim=1)
        
        train_acc += ((start_idx == data[3]) & (end_idx == data[4])).float().mean()
        pos_loss = output.loss
        
        train_loss = train_loss + pos_loss # + sel_loss
        pos_loss.backward()
        optimizer_2.step()
        scheduler_2.step()
        step_1 += 1
        
        
        if step_1 % 1200 == 0:
            tmp_train_loss = train_loss.item() / step_1
            tmp_train_acc = train_acc / step_1
            print("Epoch {} | Step {} | Position loss = {:.5f}, acc = {:.5f}".format(e,step_1,tmp_train_loss,tmp_train_acc))
            log_file.add_scalar("Training/Loss", tmp_train_loss, step_1)
            log_file.add_scalar("Training/Acc", tmp_train_acc, step_1)



def Datasets_pre_Q1(test_json_file,context):
    progress_bar = tqdm(total=(2 * len(test_json_file)))
    print("Processing Data ...")
    q_p_data = []
    for line in range(len(test_json_file)):
        q = test_json_file[line]["question"]
        ch_ids = test_json_file[line]["paragraphs"]
        q_pair = [q for id_num in range(len(ch_ids))]
        ch_p_content = [context[idx] for idx in ch_ids]
        q_p_data.append([q_pair, ch_p_content])
        progress_bar.update(1)
        
    input_ids, att_mask, token_type_ids = [], [], []
    for q_p_single in q_p_data:
        inputs = tokenizer_1(q_p_single[0], text_pair=q_p_single[1], padding="max_length", truncation=True, return_tensors="pt", max_length=384)
        input_ids.append(inputs['input_ids'].tolist())
        att_mask.append(inputs['attention_mask'].tolist())
        #token_type_ids.append(inputs['token_type_ids'].tolist())
        progress_bar.update(1)
        
    return input_ids, att_mask

def Add_Q1_Answer(test_json_file,pred_label_result):
    ch_pred = np.array(pred_label_result)
    ch_pred = list(ch_pred)
    for line in range(len(test_json_file)):
        if test_json_file[line]["relevant"] == ch_pred[line]:
            #print("True")
            test_json_file[line]["relevant"] = ch_pred[line]
        else:
            continue
    return test_json_file

def Predict_Q1(testing_loader_1, train_json_file):
    progress_bar = tqdm(total= len(testing_loader_1))
    pred_label_idx = torch.tensor([], dtype=int)
    for idx, (input_ids, att_mask) in enumerate(testing_loader_1):
        with torch.no_grad():
            test_output = model_q1(input_ids.to(device_0), attention_mask=att_mask.to(device_0))
            pred_label_idx = torch.cat((pred_label_idx, test_output[0].argmax(1).cpu().data), 0)
            #print(pred_label_idx)
        progress_bar.update(1)
    pred_label_result = []
    for line_idx in range(len(train_json_file)):
        idx = pred_label_idx[line_idx]
        pred_label_result.append(train_json_file[line_idx]["paragraphs"][idx])
        
    return pred_label_result

def Generate_Q2_Token(train_json_file):
    test_q_token = tokenizer_2([test_q["question"] for test_q in train_json_file], add_special_tokens=False) 
    test_p_token = tokenizer_2(context, add_special_tokens=False)
    
    return test_q_token, test_p_token

def End_to_end_Fine_QA(train_json_file):
    total_train_acc = 0
    input_ids, att_mask = Datasets_pre_Q1(train_json_file,context)
    torch_dataset = TensorDataset(torch.tensor(input_ids), torch.tensor(att_mask))
    testing_loader_1 = DataLoader(dataset=torch_dataset, batch_size=args.per_device_batch_size_2, shuffle=False)
    print("Data is processed successfully.")
    print("Start to choice and QA predict ...")
    test_q_token, test_p_token = Generate_Q2_Token(train_json_file)
    pred_label_result = Predict_Q1(testing_loader_1, train_json_file)
    train_json_file = Add_Q1_Answer(train_json_file, pred_label_result)
    
    context_p_token= Datasets_Q2.generate_context_token(context)
    train_q_token = Datasets_Q2.generate_token(train_json_file)
    step_1 = 0
    
    
    training_set = Datasets_Q2("train", train_json_file, train_q_token, context_p_token)
    training_loader_pre = DataLoader(training_set, batch_size=1, num_workers= args.num_worker, shuffle=True, pin_memory=True)
    
    train_loss = train_acc = 0
    model_q2.train()
    
    for data in tqdm(training_loader_pre):
        
        optimizer_2.zero_grad()
        
        data = [i.to(device_1) for i in data]
        output = model_q2(input_ids=data[0], token_type_ids=data[1], attention_mask=data[2], start_positions=data[3], end_positions=data[4])
        start_idx = torch.argmax(output.start_logits, dim=1)
        end_idx = torch.argmax(output.end_logits, dim=1)
        
        train_acc += ((start_idx == data[3]) & (end_idx == data[4])).float().mean()
        pos_loss = output.loss
        
        train_loss = train_loss + pos_loss # + sel_loss
        pos_loss.backward()
        optimizer_2.step()
        scheduler_2.step()
        step_1 += 1
        
        
        if step_1 % 1200 == 0:
            tmp_train_loss = train_loss.item() / step_1
            tmp_train_acc = train_acc / step_1
            print("Epoch {} | Step {} | Position loss = {:.5f}, acc = {:.5f}".format(e,step_1,tmp_train_loss,tmp_train_acc))
            log_file.add_scalar("Training/Loss", tmp_train_loss, step_1)
            log_file.add_scalar("Training/Acc", tmp_train_acc, step_1)
    
    total_train_acc =  train_acc / step_1
    total_train_acc_lst.append(total_train_acc.detach().cpu().numpy())
            
            
def Valid_Q2_Pos():
    print("Evaluating Validation Set ...")
    model_q2.eval() 
    with torch.no_grad():
        valid_acc = 0
        valid_loss = 0
        for i, data in enumerate(tqdm(valid_loader)):
            data = [i.to(device_1) for i in data]
            output = model_q2(input_ids=data[0], token_type_ids=data[1], attention_mask=data[2], start_positions=data[3], end_positions=data[4])
            loss = output.loss
            valid_loss += loss
            start_idx = torch.argmax(output.start_logits, dim=1)
            end_idx = torch.argmax(output.end_logits, dim=1)
            valid_acc += ((start_idx == data[3]) & (end_idx == data[4])).sum()
        tmp_valid_loss = valid_loss.item() / len(valid_loader)
        log_file.add_scalar("Validation/Loss", tmp_valid_loss, e)
        log_file.add_scalar("Validation/Exact Match", valid_acc, global_step= e)
        print("Validation | Epoch {} | acc = {:.5f}".format(e,valid_acc / (len(valid_loader) * args.per_device_batch_size_2)))
        total_valid_pos_acc_lst.append(valid_acc.detach().cpu().numpy() / (len(valid_loader) * args.per_device_batch_size_2))
        
def Valid_Q2_Par():
    model_q2.eval()
    with torch.no_grad():
        valid_acc = 0
        for i, data in enumerate(tqdm(valid_loader_test)):
            output = model_q2(input_ids=data[0].squeeze(dim=0).to(device_1), token_type_ids=data[1].squeeze(dim=0).to(device_1), attention_mask=data[2].squeeze(dim=0).to(device_1))
            valid_acc += evaluation_q2(data, output) == valid_json_file[i]["answer"]["text"]
        log_file.add_scalar("Validation_for_test/Exact Match", valid_acc, e)
        print("Validation_for_test | Epoch {} | acc = {:.5f}".format(e,valid_acc / len(valid_loader_test)))
        total_valid_par_acc_lst.append(valid_acc/ len(valid_loader_test))



tokenizer_1 = BertTokenizerFast.from_pretrained(args.tokenizer_name_1,do_lower_case=True)
tokenizer_1.save_pretrained("./tokenizer_all/q1")

#tokenizer_2 = BertTokenizerFast.from_pretrained(args.tokenizer_name_2,do_lower_case=True)
tokenizer_2 = AutoTokenizer.from_pretrained(args.tokenizer_name_2, use_fast=True)
tokenizer_2.save_pretrained("./tokenizer_all/q2")


if __name__ == '__main__':
    
    global total_train_acc_lst
    total_train_acc_lst = []
    global total_valid_pos_acc_lst
    total_valid_pos_acc_lst = []
    global total_valid_par_acc_lst
    total_valid_par_acc_lst = []
    
    random_seed = args.rand_seed
    
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    torch.manual_seed(random_seed)
    
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    '''
    #Set whether torch can use GPU
    if torch.cuda.is_available():
        device_0 = 'cuda:0'
    else: 
        device_0 = 'cpu'
    if torch.cuda.is_available():
        device_1 = 'cuda:1'
    else:
        device_1 = 'cpu'
    '''
    device_0 = device_1 = 'cuda'
        
    print("Training Device: " + device_0 + " " + device_1)
    global context, train_json_file, valid_json_file
    with open(args.context_file, newline='') as context_file:
        context = json.load(context_file)
    with open(args.train_file, newline='') as train_file:
        train_json_file = json.load(train_file)
    with open(args.valid_file, newline='') as valid_file:
        valid_json_file = json.load(valid_file)
    
    
    
    model_q1 = BertForMultipleChoice.from_pretrained(args.model_name_or_path_1)
    model_q1.to(device_0)
    
    
    model_q2 = AutoModelForQuestionAnswering.from_pretrained(args.model_name_or_path_2 , config = AutoConfig.from_pretrained(args.model_name_or_path_2))
    model_q2.to(device_1)
    
    optimizer_1 = optim.AdamW(model_q1.parameters(), lr=args.learning_rate)
    optimizer_2 = optim.AdamW(model_q2.parameters(), lr=args.learning_rate)
    
    scheduler_1 = StepLR(optimizer_1, step_size=args.gradient_accumulation_steps, gamma=args.optimizer_gamma)
    scheduler_2 = StepLR(optimizer_2, step_size=args.gradient_accumulation_steps, gamma=args.optimizer_gamma)
    
    train_labels = Generate_Labels(train_json_file)
    valid_labels = Generate_Labels(valid_json_file)
    
    
    train_dataset = Datasets_Q1(context, train_json_file, train_labels)
    valid_dataset = Datasets_Q1(context, valid_json_file, valid_labels)
    train_loader_q1 = DataLoader(dataset=train_dataset , batch_size= args.per_device_batch_size_1, shuffle=True, num_workers=args.num_worker, collate_fn=Data_Collator)
    valid_loader_q1 = DataLoader(dataset=valid_dataset , batch_size= args.per_device_batch_size_1, shuffle=True, num_workers=args.num_worker, collate_fn=Data_Collator)

    
    
    log_file = SummaryWriter("./training_log")
    valid_pos = True
    valid_par = True
    
    train_q_token = Datasets_Q2.generate_token(train_json_file)
    valid_q_token = Datasets_Q2.generate_token(valid_json_file)
    context_p_token= Datasets_Q2.generate_context_token(context)
    
    
    train_set = Datasets_Q2("train", train_json_file, train_q_token, context_p_token)
    #train_set_test = Datasets_Q2("train_test", train_json_file, train_q_token, context_p_token)
    valid_set = Datasets_Q2("valid", valid_json_file, valid_q_token, context_p_token)
    valid_set_test = Datasets_Q2("valid_test", valid_json_file, valid_q_token, context_p_token)
    
    
    train_loader = DataLoader(train_set, batch_size=args.per_device_batch_size_2, num_workers= args.num_worker, shuffle=True, pin_memory=True)
    #train_loader_test = DataLoader(train_set_test, batch_size=args.per_device_batch_size_2, num_workers= args.num_worker, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=args.per_device_batch_size_2, num_workers= args.num_worker, shuffle=False, pin_memory=True)
    valid_loader_test = DataLoader(valid_set_test, batch_size=1, shuffle=False, pin_memory=True)
    
    
    print("Start Training ...")
    
    
    for e in range(args.num_train_epochs):
        print("------------Epoch: %d ----------------" % e)
        Train_ALL()
        
        if (valid_pos == True):
            Valid_Q2_Pos()
            
        if (valid_par == True):
            Valid_Q2_Par()
    
    
    for e in range(args.num_fine_epochs):
        End_to_end_Fine_QA(train_json_file)
    
    ep = np.array([i+1 for i in range (args.num_fine_epochs)])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(ep, total_train_acc_lst, 'r-o')
    plt.savefig('train_acc_end_to_end.png')
    plt.close()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(ep, total_valid_pos_acc_lst, 'b-o')
    plt.savefig('valid_pos_acc_end_to_end.png')
    plt.close()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(ep, total_valid_par_acc_lst, 'g-o')
    plt.savefig('valid_par_acc_end_to_end.png')
    plt.close()  
    
    
    Save_model()