#!/usr/bin/env bash

cd ../

function run() {
    model=$1
    mode_path=$2
    tokenizer_path=$3
    re=$4

    python adv_summarization.py \
            --test_data_file ./dataset/code_summarization/transforms.Replace/adv_test.jsonl \
            --model_name_or_path $mode_path \
            --tokenizer_name $tokenizer_path \
            --model $model \
            --save_dir ./saved_models/ \
            --task summarization \
            --num_examples 1000 \
            --recipe $re \
            --parallel \
            2>&1 | tee ./saved_models/$model/code_summarization_$re.log
}

function codebert() {
    for attack in random textfooler pso bertattack lsh hard
    do
        run codebert microsoft/codebert-base roberta-base $attack
    done
}

function graphcodebert() {
    for attack in random textfooler pso bertattack lsh hard
    do
        run graphcodebert microsoft/graphcodebert-base microsoft/graphcodebert-base $attack
    done
}

function codet5() {
    for attack in random textfooler pso bertattack lsh hard
    do
        run codet5 Salesforce/codet5-base Salesforce/codet5-base $attack
    done
}

function plbart() {
    for attack in random textfooler pso bertattack lsh hard
    do
        run plbart ./saved_models/plbart/checkpoint_11_100000.pt ./saved_models/plbart/sentencepiece.bpe.model $attack
    done
}

# for model in codebert graphcodebert codet5 plbart
for model in plbart
do
    $model
done