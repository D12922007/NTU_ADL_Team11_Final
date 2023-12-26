import json
import argparse
import numpy as np
import csv
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset, Dataset
from transformers import AutoTokenizer, BertForQuestionAnswering, AutoModelForQuestionAnswering, BertTokenizerFast, BertTokenizer, AutoConfig
from tqdm import tqdm
import os

parser = argparse.ArgumentParser(description="Finetune a transformers model on a multiple choice task")
parser.add_argument(
    "--test_file", type=str, default="./test.json", help="A csv or a json file containing the testing data."
)
parser.add_argument(
    "--context_file", type=str, default="./context.json", help="A csv or a json file containing the context data."
)
parser.add_argument(
    "--output_file", type=str, default="./Q2_prediction.csv", help="A csv or a json file for the output file."
)
parser.add_argument(
    "--rand_seed",
    type=int,
    default=2023,
    help="Random seed value setting for training",
)
parser.add_argument(
    "--model_name_or_path",
    type=str,
    default='./q2',
    help="Path to pretrained model or model identifier from huggingface.co/models.",
    required=False,
)
parser.add_argument(
    "--max_q_seq_length",
    type=int,
    default=64,
    help=(
        "The maximum total input sequence length in question after tokenization. Sequences longer than this will be truncated,"
        " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
    ),
)
parser.add_argument(
    "--max_p_seq_length",
    type=int,
    default=445,
    help=(
        "The maximum total input sequence length in paragraph after tokenization. Sequences longer than this will be truncated,"
        " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
    ),
)
parser.add_argument(
    "--doc_stride",
    type=int,
    default=128,
    help=(
        "The maximum input stride for model to process document."
    ),
)
parser.add_argument(
    "--q1_predict_result",
    type=str,
    default='./choice_pred_test_label.npy',
    help="Path to pretrained model or model identifier from huggingface.co/models.",
    required=False,
)
parser.add_argument(
    "--tokenizer_name",
    type=str,
    default= "./tokenizer_2/",
    help="Pretrained tokenizer name or path if not the same as model_name",
)
parser.add_argument(
    "--per_device_batch_size",
    type=int,
    default=1,
    help="Batch size (per device) for the dataloader.",
)
args = parser.parse_args()


class Q2_Dataset(Dataset):

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


def evaluation(data, output):
    ans = ''
    max_p = float('-inf')
    num_of_win = data[0].shape[1]
    
    for k in range(num_of_win):
        start_p, start_idx = torch.max(output.start_logits[k], dim=0)
        #print("start_prob, start_idx",start_prob, start_idx)
        end_p, end_idx = torch.max(output.end_logits[k], dim=0)
        #print("end_prob, end_idx",end_prob, end_idx)
        p = start_p + end_p
        if start_idx > end_idx:
            continue
        if p > max_p:
            max_p = p
            ans = tokenizer.decode(data[0][0][k][start_idx:end_idx + 1])
    return ans.replace(' ','')

def Add_Q1_Answer(test_json_file):
    ch_pred = np.load(args.q1_predict_result)
    ch_pred = list(ch_pred)
    for line in range(len(test_json_file)):
        test_json_file[line]["relevant"] = ch_pred[line]
    return test_json_file

def Predict_Q2(testing_loader):
    print("Start to QA predict ...")
    predict_result = []
    model_Q2.eval()
    with torch.no_grad():
        for data in tqdm(testing_loader):
            output = model_Q2(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),attention_mask=data[2].squeeze(dim=0).to(device))
            predict_result.append(evaluation(data, output))
            
    return predict_result

def Write_Result_to_csv(predict_result):
    predict_save_file = args.output_file
    with open(predict_save_file, 'w', newline='') as out_csv:
        writer = csv.writer(out_csv)
        writer.writerow(['id', 'answer'])
        for i, test_q in enumerate(test_json_file):
            writer.writerow([test_q["id"], predict_result[i]])
    out_csv.close()
    print("Complete testing data of QA output." + " Output result is saved in: " + predict_save_file)


if __name__ == "__main__":

    # Seed
    random_seed = args.rand_seed
    
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    
    np.random.seed(random_seed)
    random.seed(random_seed)

    #Set whether torch can use GPU
    if torch.cuda.is_available():
        device = 'cuda'
    else: 
        device = 'cpu'
    print("Device: " + device)
    
    
    print("Loading testing files and processing data...")
    
    with open(args.test_file, newline='') as test_jsonfile:
        test_json_file = json.load(test_jsonfile)
    with open(args.context_file, newline='') as context_jsonfile:
        context = json.load(context_jsonfile)
    
    #tokenizer = AutoTokenizer.from_pretrained("./tokenizer/")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    #tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_name,do_lower_case=True)
    #model_Q2 = BertForQuestionAnswering.from_pretrained(args.model_name_or_path)
    model_Q2= AutoModelForQuestionAnswering.from_pretrained(args.model_name_or_path, config = AutoConfig.from_pretrained(args.model_name_or_path))
    model_Q2.to(device)

    test_q_token = tokenizer([test_q["question"] for test_q in test_json_file], add_special_tokens=False) 
    test_p_token = tokenizer(context, add_special_tokens=False)
    
    test_json_file = Add_Q1_Answer(test_json_file)
    
    testing_set = Q2_Dataset(test_json_file, test_q_token, test_p_token)
    testing_loader = DataLoader(testing_set, batch_size=args.per_device_batch_size, shuffle=False, pin_memory=True)
    print("Files are loaded completely and data is processed completely.")
    
    predict_result = Predict_Q2(testing_loader)
    Write_Result_to_csv(predict_result)