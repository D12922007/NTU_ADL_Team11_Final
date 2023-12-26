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
from transformers import AutoTokenizer, BertForMultipleChoice, BertForQuestionAnswering, BertConfig, AutoModel, BertTokenizerFast, BertTokenizer
from transformers import CONFIG_MAPPING, AutoTokenizer, BertForQuestionAnswering, BertConfig, AutoModelForQuestionAnswering, BertTokenizerFast, BertTokenizer, AutoConfig
from torch.utils.tensorboard import SummaryWriter
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
    "--model_name_or_path_1",
    type=str,
    default='./all/q1',#'deepset/roberta-base-squad2',#'hfl/chinese-macbert-large', #'hfl/chinese-bert-wwm', #'hfl/chinese-roberta-wwm-ext'
    help="Path to pretrained model or model identifier from huggingface.co/models.",
    required=False,
)
parser.add_argument(
    "--model_name_or_path_2",
    type=str,
    default='./all/q2',#'deepset/roberta-base-squad2',#'hfl/chinese-macbert-large', #'hfl/chinese-bert-wwm', #'hfl/chinese-roberta-wwm-ext'
    help="Path to pretrained model or model identifier from huggingface.co/models.",
    required=False,
)
parser.add_argument(
    "--tokenizer_name_1",
    type=str,
    default='./tokenizer_1/',#'hfl/chinese-macbert-large', #'hfl/chinese-bert-wwm', #'hfl/chinese-roberta-wwm-ext'
    help="Pretrained tokenizer name or path if not the same as model_name",
)
parser.add_argument(
    "--tokenizer_name_2",
    type=str,
    default='./tokenizer_2/',#'hfl/chinese-macbert-large', #'hfl/chinese-bert-wwm', #'hfl/chinese-roberta-wwm-ext'
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
    default=512, #384
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
args = parser.parse_args()

def End_to_end(test_json_file):
    
    input_ids, att_mask = Datasets_Q1(test_json_file,context)
    torch_dataset = TensorDataset(torch.tensor(input_ids), torch.tensor(att_mask))
    testing_loader_1 = DataLoader(dataset=torch_dataset, batch_size=args.per_device_batch_size, shuffle=False)
    print("Data is processed successfully.")
    print("Start to choice and QA predict ...")
    
    
    test_q_token, test_p_token = Generate_Q2_Token(test_json_file)
    
    pred_label_result = Predict_Q1(testing_loader_1, test_json_file)
    test_json_file = Add_Q1_Answer(test_json_file,pred_label_result)
    
    testing_set = Dataset_Q2(test_json_file, test_q_token, test_p_token)
    testing_loader_2 = DataLoader(testing_set, batch_size=1, shuffle=False, pin_memory=True)
    predict_result = Predict_Q2(testing_loader_2)
    
    
    return predict_result


def Datasets_Q1(test_json_file,context):
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

def Predict_Q1(testing_loader_1, test_json_file):
    progress_bar = tqdm(total= len(testing_loader_1))
    pred_label_idx = torch.tensor([], dtype=int)
    for idx, (input_ids, att_mask) in enumerate(testing_loader_1):
        with torch.no_grad():
            test_output = model_Q1(input_ids.to(device), attention_mask=att_mask.to(device))
            pred_label_idx = torch.cat((pred_label_idx, test_output[0].argmax(1).cpu().data), 0)
            #print(pred_label_idx)
        progress_bar.update(1)
    pred_label_result = []
    for line_idx in range(len(test_json_file)):
        idx = pred_label_idx[line_idx]
        pred_label_result.append(test_json_file[line_idx]["paragraphs"][idx])
        
    return pred_label_result


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
            ans = tokenizer_2.decode(data[0][0][k][start_idx:end_idx + 1])
    return ans.replace(' ','')

def Add_Q1_Answer(test_json_file,pred_label_result):
    ch_pred = np.array(pred_label_result)
    ch_pred = list(ch_pred)
    for line in range(len(test_json_file)):
        test_json_file[line]["relevant"] = ch_pred[line]
    return test_json_file

def Predict_Q2(testing_loader_2):
    predict_result = []
    model_Q2.eval()
    with torch.no_grad():
        for data in tqdm(testing_loader_2):
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
    
def Generate_Q2_Token(test_json_file):
    test_q_token = tokenizer_2([test_q["question"] for test_q in test_json_file], add_special_tokens=False) 
    test_p_token = tokenizer_2(context, add_special_tokens=False)
    
    return test_q_token, test_p_token


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
    print("Testing Device: " + device)


    print("Loading files ...")
    with open(args.test_file, newline='') as test_jsonfile:
        test_json_file = json.load(test_jsonfile)
    with open(args.context_file, newline='') as context_jsonfile:
        context = json.load(context_jsonfile)
        
    tokenizer_1 = BertTokenizerFast.from_pretrained(args.tokenizer_name_1, do_lower_case=True)
	#tokenizer_1 = AutoTokenizer.from_pretrained(args.tokenizer_name_1, use_fast=True)
    #tokenizer_2 = BertTokenizerFast.from_pretrained(args.tokenizer_name_2, do_lower_case=True)
    tokenizer_2 = AutoTokenizer.from_pretrained(args.tokenizer_name_2, use_fast=True)
    print("Pretrain files are loaded successfully.")
    
    model_Q1 = BertForMultipleChoice.from_pretrained(args.model_name_or_path_1)
    model_Q1.to(device)
    
    model_Q2 = AutoModelForQuestionAnswering.from_pretrained(args.model_name_or_path_2,config = AutoConfig.from_pretrained(args.model_name_or_path_2))
    model_Q2.to(device)
    
    predict_result = End_to_end(test_json_file)
    Write_Result_to_csv(predict_result)