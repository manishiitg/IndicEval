#!/bin/bash

source ./scripts/indic_eval/common_vars.sh
FOLDER_BASE=/sky-notebook/eval-results/headline


for model_name_or_path in "${model_names[@]}"; do
    model_name=${model_name_or_path##*/}
    TASK_NAME=indicheadline
    NUM_SHOTS=0short
    
    FOLDER="${FOLDER_BASE}/${TASK_NAME}/${model_name}/${NUM_SHOTS}"
    FILE=$FOLDER/metrics.json

    if echo "$model_name" | grep -qi "awq"; then
        awq_param="--awq"
    else
        awq_param=""
    fi

    echo "evaluating $model_name base on $TASK_NAME $NUM_SHOTS ..."
    if [ ! -f "$FILE" ]; then
        python3 -m eval.indicheadline.run_translate_test_eval \
        --ntrain 0 \
        --max_context_length 3750 \
        --save_dir $FOLDER \
        --model_name_or_path $model_name_or_path \
        --tokenizer_name_or_path $model_name_or_path \
        --eval_batch_size 1 \
        --use_chat_format \
        --chat_formatting_function eval.templates.create_prompt_with_chatml_format \
        $awq_param
    else
        cat "$FILE"
    fi
done