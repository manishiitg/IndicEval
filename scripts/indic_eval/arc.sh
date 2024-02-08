#!/bin/bash

# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
export CUDA_VISIBLE_DEVICES=0

model_names=(
    "manishiitg/open-aditi-hi-v2"
    "manishiitg/open-aditi-hi-v1"
)
FOLDER_BASE=/sky-notebook/eval-results

# -------------------------------------------------------------
#                       ARC-Easy
# -------------------------------------------------------------

for model_name_or_path in "${model_names[@]}"; do
    model_name=${model_name_or_path##*/}
    TASK_NAME=arc-easy-hi
    NUM_SHOTS=0short
    echo "evaluating $model_name base on $TASK_NAME $NUM_SHOTS ..."
    
    FOLDER="${FOLDER_BASE}/${TASK_NAME}/${model_name}/${NUM_SHOTS}"
    FILE=$FOLDER/metrics.json

    if [ ! -f "$FILE" ]; then
        # zero-shot
        python3 -m eval.arc.run_eval \
            --ntrain 0 \
            --dataset "ai2_arc" \
            --subset "easy" \
            --save_dir $FOLDER \
            --model_name_or_path $model_name_or_path \
            --tokenizer_name_or_path $model_name_or_path \
            --eval_batch_size 4 \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_chatml_format
    fi

    NUM_SHOTS=5short
    FOLDER="${FOLDER_BASE}/${TASK_NAME}/${model_name_or_path}/${NUM_SHOTS}"
    FILE=$FOLDER/metrics.json
    echo "evaluating $model_name base on $TASK_NAME $NUM_SHOTS ..."

    if [ ! -f "$FILE" ]; then
        # 5-shot
        python3 -m eval.arc.run_eval \
            --ntrain 5 \
            --dataset "ai2_arc" \
            --subset "easy" \
            --save_dir "/sky-notebook/eval-results/arc-easy/aditi-v2-5shot" \
            --model_name_or_path $model_name_or_path \
            --tokenizer_name_or_path $model_name_or_path \
            --eval_batch_size 1 \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_chatml_format
    fi
done

# -------------------------------------------------------------
#                       ARC-Challenge
# -------------------------------------------------------------
for model_name_or_path in "${model_names[@]}"; do
    model_name=${model_name_or_path##*/}
    TASK_NAME=arc-challenge-hi

    NUM_SHOTS=0short
    FOLDER="${FOLDER_BASE}/${TASK_NAME}/${model_name}/${NUM_SHOTS}"
    FILE=$FOLDER/metrics.json

    echo "evaluating $model_name base on $TASK_NAME $NUM_SHOTS ..."

    if [ ! -f "$FILE" ]; then
        # zero-shot
        python3 -m eval.arc.run_eval \
            --ntrain 0 \
            --dataset "ai2_arc" \
            --subset "challenge" \
            --save_dir $FOLDER \
            --model_name_or_path $model_name_or_path \
            --tokenizer_name_or_path $model_name_or_path \
            --eval_batch_size 4 \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_chatml_format
    fi
    NUM_SHOTS=5short
    FOLDER="${FOLDER_BASE}/${TASK_NAME}/${model_name_or_path}/${NUM_SHOTS}"
    FILE=$FOLDER/metrics.json
    echo "evaluating $model_name base on $TASK_NAME $NUM_SHOTS ..."

    if [ ! -f "$FILE" ]; then
        # 5-shot
        python3 -m eval.arc.run_eval \
            --ntrain 5 \
            --dataset "ai2_arc" \
            --subset "challenge" \
            --save_dir $FOLDER \
            --model_name_or_path $model_name_or_path \
            --tokenizer_name_or_path $model_name_or_path \
            --eval_batch_size 1 \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_chatml_format
    fi
done


# -------------------------------------------------------------
#                       Indic ARC-Easy
# -------------------------------------------------------------
for model_name_or_path in "${model_names[@]}"; do
    model_name=${model_name_or_path##*/}
    TASK_NAME=indic-arc-easy

    NUM_SHOTS=0short
    FOLDER="${FOLDER_BASE}/${TASK_NAME}/${model_name}/${NUM_SHOTS}"
    FILE=$FOLDER/metrics.json

    echo "evaluating $model_name base on $TASK_NAME $NUM_SHOTS ..."

    if [ ! -f "$FILE" ]; then
        # zero-shot
        python3 -m eval.arc.run_eval \
            --ntrain 0 \
            --dataset "ai4bharat/ai2_arc-hi" \
            --subset "easy" \
            --save_dir $FOLDER \
            --model_name_or_path $model_name_or_path \
            --tokenizer_name_or_path $model_name_or_path \
            --eval_batch_size 4 \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_chatml_format
    fi
    NUM_SHOTS=5short
    FOLDER="${FOLDER_BASE}/${TASK_NAME}/${model_name}/${NUM_SHOTS}"
    FILE=$FOLDER/metrics.json

    echo "evaluating $model_name base on $TASK_NAME $NUM_SHOTS ..."
    if [ ! -f "$FILE" ]; then
        # 5-shot
        python3 -m eval.arc.run_eval \
            --ntrain 5 \
            --dataset "ai4bharat/ai2_arc-hi" \
            --subset "easy" \
            --save_dir $FOLDER \
            --model_name_or_path $model_name_or_path \
            --tokenizer_name_or_path $model_name_or_path \
            --eval_batch_size 1 \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_chatml_format
    fi
done


# -------------------------------------------------------------
#                       ARC-Challenge
# -------------------------------------------------------------
for model_name_or_path in "${model_names[@]}"; do
    model_name=${model_name_or_path##*/}
    TASK_NAME=indic-arc-challenge

    NUM_SHOTS=0short
    FOLDER="${FOLDER_BASE}/${TASK_NAME}/${model_name}/${NUM_SHOTS}"
    FILE=$FOLDER/metrics.json

    echo "evaluating $model_name base on $TASK_NAME $NUM_SHOTS ..."

    # if [ ! -f "$FILE" ]; then
    #     # zero-shot
    #     python3 -m eval.arc.run_eval \
    #         --ntrain 0 \
    #         --dataset "ai4bharat/ai2_arc-hi" \
    #         --subset "challenge" \
    #         --save_dir $FOLDER \
    #         --model_name_or_path $model_name_or_path \
    #         --tokenizer_name_or_path $model_name_or_path \
    #         --eval_batch_size 4 \
    #         --use_chat_format \
    #         --chat_formatting_function eval.templates.create_prompt_with_chatml_format
    # fi
    NUM_SHOTS=5short
    FOLDER="${FOLDER_BASE}/${TASK_NAME}/${model_name}/${NUM_SHOTS}"
    FILE=$FOLDER/metrics.json

    echo "evaluating $model_name base on $TASK_NAME $NUM_SHOTS ..."
    if [ ! -f "$FILE" ]; then
        # 5-shot
        python3 -m eval.arc.run_eval \
            --ntrain 5 \
            --dataset "ai4bharat/ai2_arc-hi" \
            --subset "challenge" \
            --save_dir "/sky-notebook/eval-results/arc-challenge-hi/aditi-v2-5shot" \
            --model_name_or_path $model_name_or_path \
            --tokenizer_name_or_path $model_name_or_path \
            --eval_batch_size 1 \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_chatml_format
    fi
done