from transformers import BitsAndBytesConfig
import torch

def get_prompt(instruction: str) -> str:
    '''Format the instruction as a prompt for LLM.'''
    #print("instruction:", instruction)
    instruction=instruction.replace("下麵","下面")
    #print("instruction:", instruction)
    return f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。 USER: {instruction} ASSISTANT:"
    #return instruction

def get_prompt_few_shot(instruction: str, data: list, x: int, num: int):
    '''Format the instruction as a prompt for LLM.'''
    #print("instruction:", instruction)
    instruction=instruction.replace("下麵","下面")
    #print("instruction:", instruction)
    in_context_prompt = ''
    for i in range(num):
        in_context_prompt = in_context_prompt + " USER: {} ASSISTANT: {}".format(data[x-(i+1)]['instruction'],data[x-(i+1)]['output'])
    return f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。{in_context_prompt} USER: {instruction} ASSISTANT:"
    #return instruction

def get_bnb_config() -> BitsAndBytesConfig:
    '''Get the BitsAndBytesConfig.'''
    
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
    )
    