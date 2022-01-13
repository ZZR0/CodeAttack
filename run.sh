#!/usr/bin/env bash

function clone_bcb() {
    model=$1
    mode_path=$2
    tokenizer_path=$3
    re=$4

    python adv_clone_detection_bigclonebench.py \
            --test_data_file ./dataset/clone_detection_bigclonebench/transforms.Identifier/adv_data.jsonl \
            --model_name_or_path $mode_path \
            --tokenizer_name $tokenizer_path \
            --model $model \
            --save_dir ./saved_models/ \
            --task clone_bcb \
            --num_examples -1 \
            --recipe $re \
            --parallel \
            2>&1 | tee ./saved_models/$model/clone_detection_bigclonebench_$re_test.log
}

function clone_poj() {
    model=$1
    mode_path=$2
    tokenizer_path=$3
    re=$4

    python adv_clone_detection_poj.py \
            --test_data_file ./dataset/clone_detection_poj/transforms.Replace/adv_test.jsonl \
            --model_name_or_path $mode_path \
            --tokenizer_name $tokenizer_path \
            --model $model \
            --save_dir ./saved_models/ \
            --task clone_poj \
            --num_examples -1 \
            --recipe $re \
            --parallel \
            2>&1 | tee ./saved_models/$model/clone_detection_poj_$re.log
}

function defect() {
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

function search() {
    model=$1
    mode_path=$2
    tokenizer_path=$3
    re=$4

    python adv_search.py \
            --test_data_file ./dataset/code_search/transforms.Replace/adv_test.jsonl \
            --model_name_or_path $mode_path \
            --tokenizer_name $tokenizer_path \
            --model $model \
            --save_dir ./saved_models/ \
            --task search \
            --num_examples -1 \
            --recipe $re \
            --parallel \
            2>&1 | tee ./saved_models/$model/code_search_$re.log
}

function summarization() {
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
    task=$1
    attack=$2
    $task codebert microsoft/codebert-base roberta-base $attack
}

function graphcodebert() {
    task=$1
    attack=$2
    $task graphcodebert microsoft/graphcodebert-base microsoft/graphcodebert-base $attack
}

function codet5() {
    task=$1
    attack=$2
    $task codet5 Salesforce/codet5-base Salesforce/codet5-base $attack
}

function contracode() {
    task=$1
    attack=$2
    $task contracode ./saved_models/contracode/ckpt_transformer_hybrid_pretrain_240k.pth ./saved_models/contracode/csnjs_8k_9995p_unigram_url.model $attack
}

function plbart() {
    task=$1
    attack=$2
    $task plbart ./saved_models/plbart/checkpoint_11_100000.pt ./saved_models/plbart/sentencepiece.bpe.model $attack
}

model=$1
task=$2
attack=$3

$model $task $attack