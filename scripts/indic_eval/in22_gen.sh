#!/bin/bash

source ./scripts/indic_eval/common_vars.sh

FOLDER_BASE=/sky-notebook/eval-results/in22


for model_name_or_path in "${model_names[@]}"; do
    model_name=${model_name_or_path##*/}
    TASK_NAME=implicit_hate
    NUM_SHOTS=0short
    
    FOLDER="${FOLDER_BASE}/${TASK_NAME}/${model_name}/${NUM_SHOTS}"
    FILE=$FOLDER/metrics.json
    echo "evaluating $model_name base on $TASK_NAME $NUM_SHOTS ..."

    if echo "$model_name" | grep -qi "awq"; then
        awq_param="--awq"
    else
        awq_param=""
    fi

    if [ ! -f "$FILE" ]; then
        # zero-shot
        python3 -m eval.in22.run_eval_exact \
            --ntrain 0 \
            --dataset "ai4bharat/IN22-Gen" \
            --src_lang eng_Latn --tgt_lang hin_Deva \
            --save_dir $FOLDER \
            --model_name_or_path $model_name_or_path \
            --tokenizer_name_or_path $model_name_or_path \
            --eval_batch_size 8 \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_chatml_format \
            $awq_param
    else
        cat "$FILE"
    fi
done