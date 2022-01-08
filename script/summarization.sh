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
            --num_examples -1 \
            --recipe $re \
            2>&1 | tee ./saved_models/$model/code_summarization_$re.log
}

function codebert() {
    run codebert microsoft/codebert-base roberta-base textfooler
    run codebert microsoft/codebert-base roberta-base random
}

for model in codebert
do
    $model
done