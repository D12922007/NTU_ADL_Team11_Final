import os

ls = os.listdir("../dataset/raw_data/new_testiment")
for i in ls:
    if '_' in i: continue

    id = i.split('.')[0]
    name = i.split('(')[1].split(')')[0]
    if " " in name:
        name = name.split(" ")[1] + name.split(" ")[0]

    name = f"{id}_{name}.txt"
    print(name)
    os.rename(f"../dataset/raw_data/new_testiment/{i}", f"../dataset/raw_data/new_testiment/{name}")

ls = os.listdir("../dataset/raw_data/old_testiment")
for i in ls:
    if '_' in i: continue

    id = i[:2]
    nls = i.split(" ")
    name = nls[1] if len(nls) == 3 else nls[2]+nls[1]
    name = f"{id}_{name}.txt" if id != "22" else "22_SongOfSolomon.txt"
    print(name)
    os.rename(f"../dataset/raw_data/old_testiment/{i}", f"../dataset/raw_data/old_testiment/{name}")