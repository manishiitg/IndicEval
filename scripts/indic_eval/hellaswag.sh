



# -------------------------------------------------------------
#                       Hellaswag
# -------------------------------------------------------------

model_names=(
    "manishiitg/open-aditi-hi-v2-awq"
    "manishiitg/open-aditi-hi-v1-awq"
    "TheBloke/OpenHermes-2.5-Mistral-7B-AWQ"
    "manishiitg/open-aditi-hi-v2-dpo-awq-1.1"
)
FOLDER_BASE=/sky-notebook/eval-results/hellaswag


for model_name_or_path in "${model_names[@]}"; do
    model_name=${model_name_or_path##*/}
    TASK_NAME=hellaswag
    NUM_SHOTS=0short
    
    FOLDER="${FOLDER_BASE}/${TASK_NAME}/${model_name}/${NUM_SHOTS}"
    FILE=$FOLDER/metrics.json
    echo "evaluating $model_name base on $TASK_NAME $NUM_SHOTS ..."

    if [ ! -f "$FILE" ]; then
        python3 -m eval.hellaswag.run_eval_exact \
            --ntrain 0 \
            --save_dir $FOLDER \
            --model_name_or_path $model_name_or_path \
            --tokenizer_name_or_path $model_name_or_path \
            --eval_batch_size 4 \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_chatml_format \
            --awq \
            --use_vllm
    
    else
        cat "$FILE"

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
    echo "evaluating $model_name base on $TASK_NAME $NUM_SHOTS ..."

    if [ ! -f "$FILE" ]; then
        # zero-shot
        python3 -m eval.hellaswag.run_eval_exact \
            --ntrain 0 \
            --dataset "Thanmay/hellaswag-translated" \
            --save_dir $FOLDER \
            --model_name_or_path $model_name_or_path \
            --tokenizer_name_or_path $model_name_or_path \
            --eval_batch_size 4 \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_chatml_format \
            --awq \
            --use_vllm
    
    else
        cat "$FILE"
    fi
done