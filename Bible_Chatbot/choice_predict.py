import json
import argparse
import numpy as np
import random
import torch
import sys
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset, Dataset
from transformers import AutoTokenizer, BertForMultipleChoice, AutoModel, BertTokenizerFast, BertTokenizer
from tqdm import tqdm


parser = argparse.ArgumentParser(description="Finetune a transformers model on a multiple choice task")
parser.add_argument(
    "--test_file", type=str, default="./test.json", help="A csv or a json file containing the testing data."
)
parser.add_argument(
    "--context_file", type=str, default="./context.json", help="A csv or a json file containing the context data."
)
parser.add_argument(
    "--output_file", type=str, default="./choice_pred_test_label.npy", help="A csv or a json file containing the context data."
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
    default='./q1',
    help="Path to pretrained model or model identifier from huggingface.co/models.",
    required=False,
)
parser.add_argument(
    "--per_device_batch_size",
    type=int,
    default=1,
    help="Batch size (per device) for the dataloader.",
)
parser.add_argument(
    "--tokenizer_name",
    type=str,
    default="./tokenizer_1/",
    help="Pretrained tokenizer name or path if not the same as model_name",
)
args = parser.parse_args()



def Datasets(test_json_file,context):
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
        inputs = tokenizer(q_p_single[0], text_pair=q_p_single[1], padding="max_length", truncation=True, return_tensors="pt", max_length=384)
        input_ids.append(inputs['input_ids'].tolist())
        att_mask.append(inputs['attention_mask'].tolist())
        #token_type_ids.append(inputs['token_type_ids'].tolist())
        progress_bar.update(1)
        
    return input_ids, att_mask

def Predict_Q1(testing_loader, test_json_file):
    progress_bar = tqdm(total= len(testing_loader))
    pred_label_idx = torch.tensor([], dtype=int)
    for idx, (input_ids, att_mask) in enumerate(testing_loader):
        with torch.no_grad():
            test_output = model(input_ids.to(device), attention_mask=att_mask.to(device))
            pred_label_idx = torch.cat((pred_label_idx, test_output[0].argmax(1).cpu().data), 0)
            #print(pred_label_idx)
        progress_bar.update(1)
    pred_label_result = []
    for line_idx in range(len(test_json_file)):
        idx = pred_label_idx[line_idx]
        pred_label_result.append(test_json_file[line_idx]["paragraphs"][idx])
        
    return pred_label_result

def Write_Result_to_npy(pred_label_result):
    pred_result_file = args.output_file
    np.save(pred_result_file, np.array(pred_label_result))
    print("Testing q1 choice is completed." + " Output result is in " + pred_result_file)



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
    #tokenizer = AutoTokenizer.from_pretrained("./tokenizer/")
    #tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_name,do_lower_case=True)
    print("Pretrain files are loaded successfully.")
    
    
    
    input_ids, att_mask = Datasets(test_json_file,context)
        
    #input_ids = torch.tensor(input_ids)
    #att_mask = torch.tensor(att_mask)
    #token_type_ids = torch.tensor(token_type_ids)
    torch_dataset = TensorDataset(torch.tensor(input_ids), torch.tensor(att_mask))
    testing_loader = DataLoader(dataset=torch_dataset, batch_size=args.per_device_batch_size, shuffle=False)
    print("Data is processed successfully.")
    print("Start to q1 choice predict ...")


    #model = BertForMultipleChoice.from_pretrained("./scheduler_saved_b3_e1_choice_model_roberta-wwm_ext_model")
    model = BertForMultipleChoice.from_pretrained(args.model_name_or_path)
    model.to(device)
    
    pred_label_result = Predict_Q1(testing_loader, test_json_file)
    
    Write_Result_to_npy(pred_label_result)
    
    