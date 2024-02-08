# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
export CUDA_VISIBLE_DEVICES=0


# -------------------------------------------------------------
#                       Hellaswag
# -------------------------------------------------------------

model_names=(
    "manishiitg/open-aditi-hi-v2"
    "manishiitg/open-aditi-hi-v1"
)
FOLDER_BASE=/sky-notebook/eval-results


for model_name_or_path in "${model_names[@]}"; do
    model_name=${model_name_or_path##*/}
    TASK_NAME=hellaswag
    NUM_SHOTS=0short
    
    FOLDER="${FOLDER_BASE}/${TASK_NAME}/${model_name}/${NUM_SHOTS}"
    FILE=$FOLDER/metrics.json

    if [ ! -f "$FILE" ]; then
        python3 -m eval.hellaswag.run_eval \
            --ntrain 0 \
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

    if [ ! -f "$FILE" ]; then
        # 5-shot
        python3 -m eval.hellaswag.run_eval \
            --ntrain 5 \
            --save_dir $FOLDER \
            --model_name_or_path $model_name_or_path \
            --tokenizer_name_or_path $model_name_or_path \
            --eval_batch_size 1 \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_chatml_format
    fi
done

# -------------------------------------------------------------
#                       Indic Hellaswag
# -------------------------------------------------------------

for model_name_or_path in "${model_names[@]}"; do
    model_name=${model_name_or_path##*/}
    TASK_NAME=hellaswag-indic
    NUM_SHOTS=0short
    
    FOLDER="${FOLDER_BASE}/${TASK_NAME}/${model_name}/${NUM_SHOTS}"
    FILE=$FOLDER/metrics.json

    if [ ! -f "$FILE" ]; then
        # zero-shot
        python3 -m eval.hellaswag.run_eval \
            --ntrain 0 \
            --dataset "Thanmay/hellaswag-translated" \
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

    if [ ! -f "$FILE" ]; then
        # 5-shot
        python3 -m eval.hellaswag.run_eval \
            --ntrain 5 \
            --dataset "Thanmay/hellaswag-translated" \
            --save_dir $FOLDER \
            --model_name_or_path $model_name_or_path \
            --tokenizer_name_or_path $model_name_or_path \
            --eval_batch_size 1 \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_chatml_format
    fi
done