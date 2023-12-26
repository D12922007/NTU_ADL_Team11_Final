import json
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from transformers import AutoTokenizer, BertForQuestionAnswering, BertConfig, AutoModel, BertTokenizerFast, BertTokenizer
from transformers import CONFIG_MAPPING, AutoTokenizer, BertForQuestionAnswering, BertConfig, AutoModelForQuestionAnswering, BertTokenizerFast, BertTokenizer, AutoConfig
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
    "--model_name_or_path",
    type=str,
    default='hfl/chinese-macbert-large',#'BertConfig()',#'FlagAlpha/Llama2-Chinese-13b-Chat',#'hfl/chinese-macbert-large', #'hfl/chinese-bert-wwm', #'hfl/chinese-roberta-wwm-ext'
    help="Path to pretrained model or model identifier from huggingface.co/models.",
    required=False,
)
parser.add_argument(
    "--tokenizer_name",
    type=str,
    default='hfl/chinese-macbert-large',#'FlagAlpha/Llama2-Chinese-13b-Chat',#'hfl/chinese-macbert-large', #'hfl/chinese-bert-wwm', #'hfl/chinese-roberta-wwm-ext'
    help="Pretrained tokenizer name or path if not the same as model_name",
)
parser.add_argument(
    "--rand_seed",
    type=int,
    default=4136, #2023
    help="Random seed value setting for training",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=3e-5/2, #1.5e-5 #3e-5
    help="Initial learning rate (after the potential warmup period) to use.",
)
parser.add_argument(
    "--per_device_batch_size",
    type=int,
    default=2,
    help="Batch size (per device) for the dataloader.",
)
parser.add_argument(
    "--model_saving_path",
    type=str,
    default='./no_pretrain/q2',
    help="Path to save training model.",
    required=False,
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
parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.") #3
args = parser.parse_args()



class Datasets(Dataset):

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

        if self.split_idx == "train" or self.split_idx == "valid":
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

    def generate_token(json_file):
        return tokenizer([line["question"] for line in json_file], add_special_tokens=False)
    def generate_context_token(context_file):
        return tokenizer(context_file, add_special_tokens=False)


    
def evaluation(data, output):
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
            ans = tokenizer.decode(data[0][0][k][start_index:end_index + 1])
    
    return ans.replace(' ','')
    
def Train_Q2():
    total_train_acc = 0
    step = 0
    train_loss = train_acc = 0
    model_q2.train()
    for data in tqdm(train_loader):
        
        optimizer.zero_grad()
        
        data = [i.to(device) for i in data]
        output = model_q2(input_ids=data[0], token_type_ids=data[1], attention_mask=data[2], start_positions=data[3], end_positions=data[4])
        start_idx = torch.argmax(output.start_logits, dim=1)
        end_idx = torch.argmax(output.end_logits, dim=1)
        
        train_acc += ((start_idx == data[3]) & (end_idx == data[4])).float().mean()
        loss = output.loss
        train_loss += loss
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        step += 1

        if step % 1200 == 0:
            tmp_train_loss = train_loss.item() / step
            tmp_train_acc = train_acc / step
            print("Epoch {} | Step {} | loss = {:.5f}, acc = {:.5f}".format(e,step,tmp_train_loss,tmp_train_acc))
            log_file.add_scalar("Training/Loss", tmp_train_loss, step)
            log_file.add_scalar("Training/Acc", tmp_train_acc, step)
    
    total_train_acc = total_train_acc + train_acc / step
    total_train_acc_lst.append(total_train_acc.detach().cpu().numpy())
    
def Valid_Q2_Pos():
    print("Evaluating Validation Set ...")
    model_q2.eval()
    with torch.no_grad():
        valid_acc = 0
        valid_loss = 0
        for i, data in enumerate(tqdm(valid_loader)):
            data = [i.to(device) for i in data]
            output = model_q2(input_ids=data[0], token_type_ids=data[1], attention_mask=data[2], start_positions=data[3], end_positions=data[4])
            loss = output.loss
            valid_loss += loss
            start_idx = torch.argmax(output.start_logits, dim=1)
            end_idx = torch.argmax(output.end_logits, dim=1)
            valid_acc += ((start_idx == data[3]) & (end_idx == data[4])).sum()
        tmp_valid_loss = valid_loss.item() / len(valid_loader)
        log_file.add_scalar("Validation/Loss", tmp_valid_loss, e)
        log_file.add_scalar("Validation/Exact Match", valid_acc, global_step= e)
        print("Validation | Epoch {} | acc = {:.5f}".format(e,valid_acc / (len(valid_loader) * args.per_device_batch_size)))
        total_valid_pos_acc_lst.append(valid_acc.detach().cpu().numpy() / (len(valid_loader) * args.per_device_batch_size))
    
def Valid_Q2_Par():
    model_q2.eval()
    with torch.no_grad():
        valid_acc = 0
        for i, data in enumerate(tqdm(valid_loader_test)):
            output = model_q2(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device), attention_mask=data[2].squeeze(dim=0).to(device))
            valid_acc += evaluation(data, output) == valid_json_file[i]["answer"]["text"]
        log_file.add_scalar("Validation_for_test/Exact Match", valid_acc, e)
        print("Validation_for_test | Epoch {} | acc = {:.5f}".format(e,valid_acc / len(valid_loader_test)))
        total_valid_par_acc_lst.append(valid_acc / len(valid_loader_test))
    
if __name__ == '__main__':
    global total_train_acc_lst
    total_train_acc_lst = []
    global total_valid_pos_acc_lst
    total_valid_pos_acc_lst = []
    global total_valid_par_acc_lst
    total_valid_par_acc_lst = []
    # Random Seed
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
    print("Training Device: " + device)
    
    log_file = SummaryWriter("./training_log")
    valid_pos = True
    valid_par = True
    
    with open(args.context_file, newline='') as context_file:
        context = json.load(context_file)
        #print(type(context))
    with open(args.train_file, newline='') as train_file:
        train_json_file = json.load(train_file)
        #print(type(train_json_file))
    with open(args.valid_file, newline='') as val_file:
        valid_json_file = json.load(val_file)
        #print(type(valid_json_file))
    #with open(args.test_file, newline='') as test_file:
        #test_json_file = json.load(test_file)
        #print(type(test_json_file))
        
    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_name, do_lower_case=True)
    #tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.save_pretrained("./tokenizer_no_pretrain/")
    
    #tokenizer = AutoTokenizer.from_pretrained("./tokenizer/") #can change

    
    train_q_token = Datasets.generate_token(train_json_file)
    valid_q_token = Datasets.generate_token(valid_json_file)
    context_p_token= Datasets.generate_context_token(context)
    
    
    train_set = Datasets("train", train_json_file, train_q_token, context_p_token)
    valid_set = Datasets("valid", valid_json_file, valid_q_token, context_p_token)
    valid_set_test = Datasets("valid_test", valid_json_file, valid_q_token, context_p_token)
    
    
    train_loader = DataLoader(train_set, batch_size=args.per_device_batch_size, num_workers= args.num_worker, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=args.per_device_batch_size, num_workers= args.num_worker, shuffle=False, pin_memory=True)
    valid_loader_test = DataLoader(valid_set_test, batch_size=1, shuffle=False, pin_memory=True)
    
    
    model_q2 = BertForQuestionAnswering(BertConfig())
    model_q2.to(device)
    
    optimizer = optim.AdamW(model_q2.parameters(), lr=args.learning_rate)
    scheduler = StepLR(optimizer, step_size=args.gradient_accumulation_steps, gamma=args.optimizer_gamma)
    
    
    print("Start Training ...")
    
    for e in range(args.num_train_epochs):
        print("------------Epoch: %d ----------------" % e)
        Train_Q2()
    
        if (valid_pos == True):
            Valid_Q2_Pos()
            
        if (valid_par == True):
            Valid_Q2_Par()
    
    print("total_train_acc_lst",total_train_acc_lst)
    print("total_valid_pos_acc_lst",total_valid_pos_acc_lst)
    print("total_valid_par_acc_lst",total_valid_par_acc_lst)
    
    ep = np.array([i+1 for i in range (args.num_train_epochs)])
    plt.xlabel('Accuracy')
    plt.ylabel('Epoch')
    plt.plot(ep, np.array(total_train_acc_lst), 'r-o')
    plt.savefig('train_acc_no_pretrain.png')
    plt.close()
    plt.xlabel('Accuracy')
    plt.ylabel('Epoch')
    plt.plot(ep, np.array(total_valid_pos_acc_lst), 'b-o')
    plt.savefig('valid_pos_acc_no_pretrain.png')
    plt.close()
    plt.xlabel('Accuracy')
    plt.ylabel('Epoch')
    plt.plot(ep, np.array(total_valid_par_acc_lst), 'g-o')
    plt.savefig('valid_par_acc_no_pretrain.png')
    plt.close()  
    
    
    model_save_dir = args.model_saving_path
    model_q2.save_pretrained(model_save_dir)
    
    #紀錄個基準loss = 0.068,acc = 0.965, val_for_test acc = 0.775
    