#!/usr/bin/env bash

cd ../

function run() {
    model=$1
    mode_path=$2
    tokenizer_path=$3

    python adv_clone_detection_poj.py \
            --test_data_file ./dataset/clone_detection_poj/transforms.Replace/adv_test.jsonl \
            --model_name_or_path $mode_path \
            --tokenizer_name $tokenizer_path \
            --model $model \
            --save_dir ./saved_models/ \
            --task clone_poj \
            --num_examples -1 \
            2>&1 | tee ./saved_models/$model/clone_detection_poj.log
}

function codebert() {
    run codebert microsoft/codebert-base roberta-base
}

for model in codebert
do
    $model
done