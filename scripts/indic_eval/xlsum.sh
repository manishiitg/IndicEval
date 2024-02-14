#!/bin/bash

source ./common_vars.sh
FOLDER_BASE=/sky-notebook/eval-results


for model_name_or_path in "${model_names[@]}"; do
    model_name=${model_name_or_path##*/}
    TASK_NAME=xlsum
    NUM_SHOTS=0short
    
    FOLDER="${FOLDER_BASE}/${TASK_NAME}/${model_name}/${NUM_SHOTS}"
    FILE=$FOLDER/metrics.json
    echo "evaluating $model_name base on $TASK_NAME $NUM_SHOTS ..."

    if echo "$model_name" | grep -qi "awq"; then
        awq_param="--awq"
    else
        awq_param=""

    if [ ! -f "$FILE" ]; then
        # 1-shot
        python3 -m eval.xlsum.run_eval \
            --ntrain 1 \
            --max_context_length 3750 \
            --save_dir $FOLDER \
            --model_name_or_path $model_name_or_path \
            --tokenizer_name_or_path $model_name_or_path \
            --eval_batch_size 1 \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_chatml_format \
            $awq_param