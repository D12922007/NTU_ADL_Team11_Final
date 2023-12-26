import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import json
from peft import PeftModel
from utils import get_prompt, get_bnb_config, get_prompt_few_shot
import argparse

def perplexity(
    model, tokenizer, data, max_length=2048,
):
    data_size = len(data)
    instructions = [get_prompt_few_shot(data[x]["instruction"], data, x, args.num_of_in_context) for x in range(data_size)]
    outputs = [x["output"] for x in data]

    # Tokenize data
    tokenized_instructions = tokenizer(instructions, add_special_tokens=False)
    tokenized_outputs = tokenizer(outputs, add_special_tokens=False)
    output_masks = []

    # Format data
    for i in tqdm(range(data_size)):
        instruction_input_ids = [tokenizer.bos_token_id] + \
            tokenized_instructions["input_ids"][i]
        output_input_ids = tokenized_outputs["input_ids"][i] + \
            [tokenizer.eos_token_id]
        tokenized_instructions["input_ids"][i] = instruction_input_ids + \
            output_input_ids
        tokenized_instructions["attention_mask"][i] = [
            1] * len(tokenized_instructions["input_ids"][i])
        output_mask = [0] * len(instruction_input_ids) + \
            [1] * len(output_input_ids)

        tokenized_instructions["input_ids"][i] = torch.tensor(
            tokenized_instructions["input_ids"][i][:max_length])
        tokenized_instructions["attention_mask"][i] = torch.tensor(
            tokenized_instructions["attention_mask"][i][:max_length])
        output_mask = torch.tensor(output_mask[:max_length])
        output_masks.append(output_mask)

    # Calculate ppl
    ppls = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    for i in tqdm(range(data_size)):
        input_ids = tokenized_instructions["input_ids"][i].unsqueeze(0)
        attn_mask = tokenized_instructions["attention_mask"][i].unsqueeze(0)
        output_mask = output_masks[i].unsqueeze(0)
        label = input_ids

        with torch.no_grad():
            out_logits = model(input_ids, attention_mask=attn_mask).logits
            out = model.generate(input_ids = input_ids, attention_mask=attn_mask)
            result = tokenizer.batch_decode(out , skip_special_tokens=True)
            print("result ",result)
            print("result_start ",result[0].find('ASSISTANT'))
            print("result_start ",result[0][result[0].rfind('ASSISTANT')+11:])
            data[i]['output']=result[0][result[0].find('ASSISTANT')+11:]
        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_label = label[..., 1:].contiguous()
        shift_output_mask = output_mask[..., 1:].contiguous()
        perplexity_batch = torch.exp(
            (loss_fct(shift_logits.transpose(1, 2),
             shift_label) * shift_output_mask).sum(1)
            / shift_output_mask.sum(1)
        )
        ppls += perplexity_batch.tolist()
    return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}, data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="./saved_model_1/",
        required=False,
        help="Path to the checkpoint of Taiwan-LLM-7B-v2.0-chat. If not set, this script will use "
        "the checkpoint from Huggingface (revision = 5073b2bbc1aa5519acdc865e99832857ef47f7c9)."
    )
    parser.add_argument(
        "--peft_path",
        type=str,
        required=False,
        default="./adapter_checkpoint_1",
        help="Path to the saved PEFT checkpoint."
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="./data/public_test.json",
        required=False,
        help="Path to test data."
    )
    parser.add_argument(
        "--num_of_in_context",
        type=int,
        default="10",
        required=False,
        help="Path to test data."
    )
	parser.add_argument(
        "--output_data_path",
        type=str,
        default="'./output_bonus_result_public_few.json'",
        required=False,
        help="Path to output data."
    )
    parser.add_argument(
        "--save_pretrain",
        type=bool,
        default=False,
        required=False,
        help="Save pretrain model or not."
    )
    args = parser.parse_args()

    # Load model
    bnb_config = get_bnb_config()

    if not (args.save_pretrain):
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            cache_dir=None,
            load_in_4bit=False,
            load_in_8bit=True,
            device_map="cuda:0",
            quantization_config=bnb_config,
            config = AutoConfig.from_pretrained(args.base_model_path),
            torch_dtype=torch.float32,
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #model.to(device)
        print('Using device:', device)
        print()
        #Additional Info when using cuda
        if device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
        '''
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config
        )
        '''
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    else:
        j = 0
        for model_path in['FlagAlpha/Llama2-Chinese-13b-Chat', 'FlagAlpha/Llama2-Chinese-7b-Chat', 'bigscience/bloomz-7b1-mt', 'bigscience/bloomz-7b1-p3']:
            j = j+1
            model = AutoModelForCausalLM.from_pretrained(model_path, config = AutoConfig.from_pretrained(model_path))
            model.save_pretrained('./saved_model_{}/'.format(j))
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            tokenizer.save_pretrained('./saved_model_{}/'.format(j))

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load LoRA
    #model = PeftModel.from_pretrained(model, args.peft_path)
    #model.to(device)
    with open(args.test_data_path, "r") as f:
        data = json.load(f)
    
    model.eval()
    ppl, data = perplexity(model, tokenizer, data)
    print("Mean perplexity:", ppl["mean_perplexity"])
    with open(args.output_data_path, 'w', encoding = "utf-8") as out_file:
        out_file.write('\ufeff')
        out_file.write("[\n")
        for i in range(len(data)):
            if i != len(data)-1 :
                line = "  {\n    \"id\": \"" + data[i]['id'] + "\",\n    \"output\": \"" + data[i]["output"] + "\"\n  },"
                out_file.write(line)
                out_file.write("\n")
            else:
                line = "  {\n    \"id\": \"" + data[i]['id'] + "\",\n    \"output\": \"" + data[i]["output"] + "\"\n  }"
                out_file.write(line)
                out_file.write("\n")
        out_file.write("]")
    out_file.close()