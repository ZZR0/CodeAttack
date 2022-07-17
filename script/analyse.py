import os
import re
import numpy as np

def get_value(s):
    s = s.replace("[", "").replace("]", "").replace("(", "").replace(")", "").replace("%", "")
    return int(s)

def get_score(line):
    value = line.split(" --> ")[0]
    label = value.split(" ")[0]
    condf = value.split(" ")[1]

    condf = get_value(condf)
    label = 1 if condf >= 50 else 0

    socre = 100 - abs(label*100 - condf)

    return socre, label, condf

def get_confident(model, method, task):
    file_path = f"../saved_models/{model}/{task}_{method}.log"
    cmd = f"cat {file_path} | grep '\-\->'"
    result = os.popen(cmd).readlines()
    values, labels, condfs = [], [], []
    for line in result:
        if not "-->" in line: continue
        if line.startswith("[["):
            line = line.strip()
        else:
            line = line.split("][[")[1]

        if "Socre:" in line or "[[[FAILED]]]" in line or "[[[SKIPPED]]]" in line:
            score, label, condf = get_score(line)
            values.append(score)
            labels.append(label)
            condfs.append(condf)
    
    assert len(values) == 2690
    return values, labels, condfs

def str_hash(s):
    pattern = re.compile(r"\W|\s|\d")
    s = re.sub(pattern, "", s)
    return hash(s)

def get_attack_socre(line):
    value = line.split(" --> ")[1]
    label = value.split(" ")[0]
    condf = value.split(" ")[1]

    condf = get_value(condf)
    label = 1 if condf >= 50 else 0

    socre = 100 - abs(label*100 - condf)

    return socre, label, condf

def process_example(example):
    type = "failed"
    if "Socre" in example[1]: type = "succeeded"
    elif "SKIPPED" in example[1]: type = "skipped"
    elif "FAILED" in example[1]: type = "failed"
    socre, label, condf = get_score(example[1])
    id = str_hash(example[2])

    if type !=  "succeeded":
        # return {"id": id, "type": type, "socre": socre, "label": label, "condf": condf, "ori":example}
        return {"id": id, "type": type, "socre": socre, "label": label, "condf": condf}
    else:
        adv_socre, adv_label, adv_condf = get_attack_socre(example[1])
        return {"id": id, "type": type, "socre": socre, "label": label, "condf": condf, 
                "adv_socre": adv_socre, "adv_label": adv_label, "adv_condf":adv_condf}


def get_examples(model, method, task):
    file_path = f"../saved_models/{model}/{task}_{method}.log"
    with open(file_path, "r") as f:
        data = f.read()
    
    pattern = re.compile(r"\[Succeeded / Failed / Skipped / Total\].*it/s\]")
    data = re.sub(pattern, "", data)
    pattern = re.compile(r"\[Succeeded / Failed / Skipped / Total\].*s/it\]")
    data = re.sub(pattern, "", data)
    lines = [line.strip() for line in data.split("\n") if line.strip()]
    
    # data = "\n".join(lines)
    # print(data)
    examples = []
    idx = 0
    while idx < len(lines):
        if lines[idx].startswith("---------------------------------------------"):
            example = [lines[idx]]
            _idx = idx+1
            while _idx < len(lines) and not lines[_idx].startswith("---------------------------------------------"):
                if lines[_idx].startswith("["):
                    example.append(lines[_idx])
                _idx += 1
            examples.append(example)
            idx = _idx
        else:
            idx += 1

    examples = [process_example(example) for example in examples]
    assert len(examples) == 2690

    dict_examples = {}
    for example in examples:
        dict_examples[example["id"]] = example
    return dict_examples

def analyse(examples0, examples1):
    count = 0 
    socres0, socres1 = [], []
    for example0 in examples0.values():
        if not example0["id"] in examples1: continue
        example1 = examples1[example0["id"]]
        if example0["type"] == "failed" and example1["type"] == "succeeded":
            socres0.append(example0["socre"])
            socres1.append(example1["socre"])
            print(example0)
            print(example1)
            print()

        count += 1
    
    print(count)
    print(len(socres0))
    print("socre0: {}".format(np.mean(socres0)))
    print("socre1: {}".format(np.mean(socres1)))


if __name__ == "__main__":
    models = ["codebert", "graphcodebert"]
    methods = ["random"]
    tasks = ["defect_detection"]

    queries_result, result, bleu_result = {}, {}, {}

    scores0, labels0, condfs0 = get_confident(models[0], methods[0], tasks[0])
    print(models[0], np.mean(scores0))
    scores1, labels1, condfs1 = get_confident(models[1], methods[0], tasks[0])
    print(models[1], np.mean(scores1))

    a = np.array(labels0) == np.array(labels1)
    print(a.sum())

    examples0 = get_examples(models[0], methods[0], tasks[0])
    examples1 = get_examples(models[1], methods[0], tasks[0])

    analyse(examples0, examples1)

