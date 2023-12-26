import os
from tqdm import tqdm
import json

bible = []
bible_testiment = ["old_testiment", "new_testiment"]
for testiment in bible_testiment:
    file_list = sorted(os.listdir(f"../dataset/raw_data/{testiment}"))
    for file in tqdm(file_list):
        with open(f"./raw_data/{testiment}/{file}", "r") as f:
            lines = list(filter(lambda x: x[0].isdigit(), f.readlines()))
            for line in lines:
                line_ls = line.split(" ")
                id = line_ls[0]
                context = "".join(line_ls[1:]).replace('\n', '')
                bible.append({"id":f"{file.replace('.txt', '')}:{id}", "context":context})

with open("../dataset/processed_data/raw_bible.jsonl", "w") as f:
    for scripture in bible:
        json.dump(scripture, f)
        f.write("\n")