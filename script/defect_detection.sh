#!/usr/bin/env bash

cd ../

function run() {
    model=$1
    mode_path=$2
    tokenizer_path=$3
    re=$4

    python adv_defect_detection.py \
            --test_data_file ./dataset/defect_detection/transforms.Replace/adv_test.jsonl \
            --model_name_or_path $mode_path \
            --tokenizer_name $tokenizer_path \
            --model $model \
            --save_dir ./saved_models/ \
            --task defect \
            --num_examples -1 \
            --recipe $re \
            --parallel \
            2>&1 | tee ./saved_models/$model/defect_detection_$re.log
}

function codebert() {
    run codebert microsoft/codebert-base roberta-base textfooler
    run codebert microsoft/codebert-base roberta-base random
}

function graphcodebert() {
    run graphcodebert microsoft/graphcodebert-base microsoft/graphcodebert-base textfooler
    run graphcodebert microsoft/graphcodebert-base microsoft/graphcodebert-base random
}

function codet5() {
    run codet5 Salesforce/codet5-base Salesforce/codet5-base textfooler
    run codet5 Salesforce/codet5-base Salesforce/codet5-base random
}

function plbart() {
    run plbart ./saved_models/plbart/checkpoint_11_100000.pt ./saved_models/plbart/sentencepiece.bpe.model textfooler
    run plbart ./saved_models/plbart/checkpoint_11_100000.pt ./saved_models/plbart/sentencepiece.bpe.model random
}

# for model in codebert graphcodebert codet5 plbart
for model in plbart
do
    $model
done