#!/bin/bash

source ./common_vars.sh
FOLDER_BASE=/sky-notebook/eval-results/indicqa


for model_name_or_path in "${model_names[@]}"; do
    model_name=${model_name_or_path##*/}
    TASK_NAME=indicqa
    NUM_SHOTS=no-context

    NUM_SHOTS=with-context
    FOLDER="${FOLDER_BASE}/${TASK_NAME}/${model_name}/${NUM_SHOTS}"
    FILE=$FOLDER/metrics.json
    echo "evaluating $model_name base on $TASK_NAME $NUM_SHOTS ..."

    if echo "$model_name" | grep -qi "awq"; then
        awq_param="--awq"
    else
        awq_param=""

    if [ ! -f "$FILE" ]; then
        # with context
        python3 -m eval.indicqa.run_translate_test_eval \
            --ntrain 0 \
            --max_context_length 3750 \
            --save_dir $FOLDER \
            --model_name_or_path $model_name_or_path \
            --tokenizer_name_or_path $model_name_or_path \
            --eval_batch_size 2 \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_chatml_format \
            $awq_param
    else
        cat "$FILE"
    fi
done
