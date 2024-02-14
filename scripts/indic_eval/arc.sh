#!/bin/bash

source ./common_vars.sh

FOLDER_BASE=/sky-notebook/eval-results/arc
# -------------------------------------------------------------
#                       ARC-Easy
# -------------------------------------------------------------

for model_name_or_path in "${model_names[@]}"; do
    model_name=${model_name_or_path##*/}
    TASK_NAME=arc-easy-exact
    NUM_SHOTS=0short
    echo "evaluating $model_name base on $TASK_NAME $NUM_SHOTS ..."
    
    FOLDER="${FOLDER_BASE}/${TASK_NAME}/${model_name}/${NUM_SHOTS}"
    FILE=$FOLDER/metrics.json

    if echo "$model_name" | grep -qi "awq"; then
        awq_param="--awq"
    else
        awq_param=""

    if [ ! -f "$FILE" ]; then
        # zero-shot
        python3 -m eval.arc.run_eval_exact \
            --ntrain 0 \
            --dataset "ai2_arc" \
            --subset "easy" \
            --save_dir $FOLDER \
            --model_name_or_path $model_name_or_path \
            --tokenizer_name_or_path $model_name_or_path \
            --eval_batch_size 4 \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_chatml_format \
            $awq_param
    else
        cat "$FILE"
    fi 
done


# -------------------------------------------------------------
#                       ARC-Easy
# -------------------------------------------------------------

# for model_name_or_path in "${model_names[@]}"; do
#     model_name=${model_name_or_path##*/}
#     TASK_NAME=arc-easy
#     NUM_SHOTS=0short
#     echo "evaluating $model_name base on $TASK_NAME $NUM_SHOTS ..."
    
#     FOLDER="${FOLDER_BASE}/${TASK_NAME}/${model_name}/${NUM_SHOTS}"
#     FILE=$FOLDER/metrics.json

#     if [ ! -f "$FILE" ]; then
#         # zero-shot
#         python3 -m eval.arc.run_eval \
#             --ntrain 0 \
#             --dataset "ai2_arc" \
#             --subset "easy" \
#             --save_dir $FOLDER \
#             --model_name_or_path $model_name_or_path \
#             --tokenizer_name_or_path $model_name_or_path \
#             --eval_batch_size 4 \
#             --use_chat_format \
#             --chat_formatting_function eval.templates.create_prompt_with_chatml_format
#     fi
# done

# -------------------------------------------------------------
#                       ARC-Challenge
# -------------------------------------------------------------
for model_name_or_path in "${model_names[@]}"; do
    model_name=${model_name_or_path##*/}
    TASK_NAME=arc-challenge-exact

    NUM_SHOTS=0short
    FOLDER="${FOLDER_BASE}/${TASK_NAME}/${model_name}/${NUM_SHOTS}"
    FILE=$FOLDER/metrics.json

    echo "evaluating $model_name base on $TASK_NAME $NUM_SHOTS ..."
    if echo "$model_name" | grep -qi "awq"; then
        awq_param="--awq"
    else
        awq_param=""

    if [ ! -f "$FILE" ]; then
        # zero-shot
        python3 -m eval.arc.run_eval_exact \
            --ntrain 0 \
            --dataset "ai2_arc" \
            --subset "challenge" \
            --save_dir $FOLDER \
            --model_name_or_path $model_name_or_path \
            --tokenizer_name_or_path $model_name_or_path \
            --eval_batch_size 4 \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_chatml_format \
            $awq_param
    else
        cat "$FILE"
    fi
done

# -------------------------------------------------------------
#                       ARC-Challenge
# -------------------------------------------------------------
# for model_name_or_path in "${model_names[@]}"; do
#     model_name=${model_name_or_path##*/}
#     TASK_NAME=arc-challenge

#     NUM_SHOTS=0short
#     FOLDER="${FOLDER_BASE}/${TASK_NAME}/${model_name}/${NUM_SHOTS}"
#     FILE=$FOLDER/metrics.json

#     echo "evaluating $model_name base on $TASK_NAME $NUM_SHOTS ..."

#     if [ ! -f "$FILE" ]; then
#         # zero-shot
#         python3 -m eval.arc.run_eval \
#             --ntrain 0 \
#             --dataset "ai2_arc" \
#             --subset "challenge" \
#             --save_dir $FOLDER \
#             --model_name_or_path $model_name_or_path \
#             --tokenizer_name_or_path $model_name_or_path \
#             --eval_batch_size 4 \
#             --use_chat_format \
#             --chat_formatting_function eval.templates.create_prompt_with_chatml_format
#     fi
# done

# -------------------------------------------------------------
#                       Indic ARC-Easy
# -------------------------------------------------------------
for model_name_or_path in "${model_names[@]}"; do
    model_name=${model_name_or_path##*/}
    TASK_NAME=indic-arc-easy-exact

    NUM_SHOTS=0short
    FOLDER="${FOLDER_BASE}/${TASK_NAME}/${model_name}/${NUM_SHOTS}"
    FILE=$FOLDER/metrics.json

    echo "evaluating $model_name base on $TASK_NAME $NUM_SHOTS ..."

    if echo "$model_name" | grep -qi "awq"; then
        awq_param="--awq"
    else
        awq_param=""

    if [ ! -f "$FILE" ]; then
        # zero-shot
        python3 -m eval.arc.run_eval_exact \
            --ntrain 0 \
            --dataset "ai4bharat/ai2_arc-hi" \
            --subset "easy" \
            --save_dir $FOLDER \
            --model_name_or_path $model_name_or_path \
            --tokenizer_name_or_path $model_name_or_path \
            --eval_batch_size 4 \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_chatml_format \
            $awq_param
    else
        cat "$FILE"
    fi
done



# -------------------------------------------------------------
#                       Indic ARC-Easy
# -------------------------------------------------------------
# for model_name_or_path in "${model_names[@]}"; do
#     model_name=${model_name_or_path##*/}
#     TASK_NAME=indic-arc-easy

#     NUM_SHOTS=0short
#     FOLDER="${FOLDER_BASE}/${TASK_NAME}/${model_name}/${NUM_SHOTS}"
#     FILE=$FOLDER/metrics.json

#     echo "evaluating $model_name base on $TASK_NAME $NUM_SHOTS ..."

#     if [ ! -f "$FILE" ]; then
#         # zero-shot
#         python3 -m eval.arc.run_eval \
#             --ntrain 0 \
#             --dataset "ai4bharat/ai2_arc-hi" \
#             --subset "easy" \
#             --save_dir $FOLDER \
#             --model_name_or_path $model_name_or_path \
#             --tokenizer_name_or_path $model_name_or_path \
#             --eval_batch_size 4 \
#             --use_chat_format \
#             --chat_formatting_function eval.templates.create_prompt_with_chatml_format
#     fi
# done

# -------------------------------------------------------------
#                       ARC-Challenge
# -------------------------------------------------------------
for model_name_or_path in "${model_names[@]}"; do
    model_name=${model_name_or_path##*/}
    TASK_NAME=indic-arc-challenge-exact

    NUM_SHOTS=0short
    FOLDER="${FOLDER_BASE}/${TASK_NAME}/${model_name}/${NUM_SHOTS}"
    FILE=$FOLDER/metrics.json

    echo "evaluating $model_name base on $TASK_NAME $NUM_SHOTS ..."

    if echo "$model_name" | grep -qi "awq"; then
        awq_param="--awq"
    else
        awq_param=""

    if [ ! -f "$FILE" ]; then
        # zero-shot
        python3 -m eval.arc.run_eval_exact \
            --ntrain 0 \
            --dataset "ai4bharat/ai2_arc-hi" \
            --subset "challenge" \
            --save_dir $FOLDER \
            --model_name_or_path $model_name_or_path \
            --tokenizer_name_or_path $model_name_or_path \
            --eval_batch_size 4 \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_chatml_format \
            $awq_param
    else
        cat "$FILE"
    fi
done


# -------------------------------------------------------------
#                       ARC-Challenge
# -------------------------------------------------------------
# for model_name_or_path in "${model_names[@]}"; do
#     model_name=${model_name_or_path##*/}
#     TASK_NAME=indic-arc-challenge

#     NUM_SHOTS=0short
#     FOLDER="${FOLDER_BASE}/${TASK_NAME}/${model_name}/${NUM_SHOTS}"
#     FILE=$FOLDER/metrics.json

#     echo "evaluating $model_name base on $TASK_NAME $NUM_SHOTS ..."

#     if [ ! -f "$FILE" ]; then
#         # zero-shot
#         python3 -m eval.arc.run_eval \
#             --ntrain 0 \
#             --dataset "ai4bharat/ai2_arc-hi" \
#             --subset "challenge" \
#             --save_dir $FOLDER \
#             --model_name_or_path $model_name_or_path \
#             --tokenizer_name_or_path $model_name_or_path \
#             --eval_batch_size 4 \
#             --use_chat_format \
#             --chat_formatting_function eval.templates.create_prompt_with_chatml_format
#     fi
# done