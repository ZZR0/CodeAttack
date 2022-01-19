import os
import numpy as np

def get_queries(model, method, task):
    file_path = f"../saved_models/{model}/{task}_{method}.log"
    cmd = f"cat {file_path} | grep '| Avg num queries:'"
    result = os.popen(cmd).read()

    num = 0
    if "queries" in result:
        num = float(result.strip().split("|")[2])
    return num

def get_acc(model, method, task):
    file_path = f"../saved_models/{model}/{task}_{method}.log"
    cmd = f"cat {file_path} | grep '| Accuracy under attack:'"
    result = os.popen(cmd).read()

    num = 0
    if "Accuracy" in result:
        num = float(result.strip().split("|")[2].replace("%","")) / 100
    return num

def get_bleu(model, method, task):
    file_path = f"../saved_models/{model}/{task}_{method}.log"
    cmd = f"cat {file_path} | grep '| Perturbed BLEU-4:'"
    result = os.popen(cmd).read()

    num = 0
    if "BLEU" in result:
        num = float(result.strip().split("|")[2])
    return num

def write_csv(result, file_path="result.csv"):
    tasks = list(result.keys())
    models = list(result[tasks[0]].keys())
    methods = list(result[tasks[0]][models[0]].keys())

    with open(file_path, "w") as f:
        for task in tasks:
            for method in methods:
                f.write(f",{task}")
        f.write("\n")
        for task in tasks:
            for method in methods:
                f.write(f",{method}")
        f.write("\n")
        
        for model in models:
            f.write(f"{model}")
            for task in tasks:
                for method in methods:
                    num = result[task][model][method]
                    f.write(f",{num}")
            f.write("\n")
        
        f.write("avg")
        for task in tasks:
            for method in methods:
                avg_num = []
                for model in models:
                    num = result[task][model][method]
                    if num > 0:
                        avg_num.append(num)
                avg_num = round(np.mean(avg_num), 2)
                f.write(f",{avg_num}")
        f.write("\n")


if __name__ == "__main__":
    models = ["codebert", "codegpt", "codet5", "codetrans", "contracode", "cotext", "graphcodebert", "plbart"]
    methods = ["random", "textfooler", "bae", "bertattack", "lsh", "hard", "grandom", "random+"]
    tasks = ["defect_detection", "code_summarization"]

    queries_result, result, bleu_result = {}, {}, {}

    for task in tasks:
        queries_result[task] = {}
        result[task] = {}

        for model in models:
            queries_result[task][model] = {}
            result[task][model] = {}

            for method in methods:
                num = get_queries(model, method, task)
                acc = get_acc(model, method, task)
                bleu = get_bleu(model, method, task)

                queries_result[task][model][method] = num
                result[task][model][method] = acc if task == "defect_detection" else bleu

    
    write_csv(queries_result, file_path="queries_result.csv")
    write_csv(result)
