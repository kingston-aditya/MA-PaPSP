import json
import os

PATH = "/data/datasets/ucf101/"

def make_retrievals(K):
    # get cat names
    f1 = open(os.path.join(PATH, "split_zhou_UCF101.json"))
    js_catg = json.load(f1)
    f1.close()

    fin_out = {}
    tag = set()
    for i in range(len(js_catg['train'])):
        if js_catg['train'][i][-1] not in tag:
            fin_out[js_catg['train'][i][-1]] = [js_catg['train'][i][0]]
            tag.add(js_catg['train'][i][-1])
        elif len(fin_out[js_catg['train'][i][-1]])<K:
            fin_out[js_catg['train'][i][-1]].append(js_catg['train'][i][0])
        else:
            pass
        tag = set(tag)

    a = {"train":[]}
    lst = list(fin_out.keys())
    for i in lst:
        for j in fin_out[i]:
            a["train"].append([j, i])

    
    with open(os.path.join(PATH, "zhou_pets_split_{}.json".format(str(K))), "w") as fp:
        json.dump(a, fp) 

for i in [5,10,15,20,25]:
    make_retrievals(i)


