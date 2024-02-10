# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
export CUDA_VISIBLE_DEVICES=0

model_names=(
    "manishiitg/open-aditi-hi-v2-awq"
    "manishiitg/open-aditi-hi-v1-awq"
)
FOLDER_BASE=/sky-notebook/eval-results


# -------------------------------------------------------------
#                       Indic BoolQ
# -------------------------------------------------------------

for model_name_or_path in "${model_names[@]}"; do
    model_name=${model_name_or_path##*/}
    TASK_NAME=boolq-hi-exact
    NUM_SHOTS=0short
    
    FOLDER="${FOLDER_BASE}/${TASK_NAME}/${model_name}/${NUM_SHOTS}"
    FILE=$FOLDER/metrics.json
    echo "evaluating $model_name base on $TASK_NAME $NUM_SHOTS ..."

    if [ ! -f "$FILE" ]; then
        # zero-shot
        python3 -m eval.boolq.run_translated_eval_exact \
            --ntrain 0 \
            --save_dir $FOLDER \
            --model_name_or_path $model_name_or_path \
            --tokenizer_name_or_path $model_name_or_path \
            --eval_batch_size 4 \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_chatml_format \
            --use_vllm \
            --awq

    fi
done

# -------------------------------------------------------------
#                       Indic BoolQ
# -------------------------------------------------------------

for model_name_or_path in "${model_names[@]}"; do
    model_name=${model_name_or_path##*/}
    TASK_NAME=boolq-hi
    NUM_SHOTS=0short
    
    FOLDER="${FOLDER_BASE}/${TASK_NAME}/${model_name}/${NUM_SHOTS}"
    FILE=$FOLDER/metrics.json
    echo "evaluating $model_name base on $TASK_NAME $NUM_SHOTS ..."

    if [ ! -f "$FILE" ]; then
        # zero-shot
        python3 -m eval.boolq.run_translated_eval \
            --ntrain 0 \
            --save_dir $FOLDER \
            --model_name_or_path $model_name_or_path \
            --tokenizer_name_or_path $model_name_or_path \
            --eval_batch_size 4 \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_chatml_format \
            --awq

    fi
done

# -------------------------------------------------------------
#                            BoolQ
# -------------------------------------------------------------

for model_name_or_path in "${model_names[@]}"; do
    model_name=${model_name_or_path##*/}
    TASK_NAME=boolq-exact
    NUM_SHOTS=0short
    
    FOLDER="${FOLDER_BASE}/${TASK_NAME}/${model_name}/${NUM_SHOTS}"
    FILE=$FOLDER/metrics.json
    echo "evaluating $model_name base on $TASK_NAME $NUM_SHOTS ..."

    if [ ! -f "$FILE" ]; then
        # zero-shot
        python3 -m eval.boolq.run_eval_exact \
            --ntrain 0 \
            --save_dir $FOLDER \
            --model_name_or_path $model_name_or_path \
            --tokenizer_name_or_path $model_name_or_path \
            --eval_batch_size 4 \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_chatml_format \
            --use_vllm \
            --awq

    fi
done

for model_name_or_path in "${model_names[@]}"; do
    model_name=${model_name_or_path##*/}
    TASK_NAME=boolq
    NUM_SHOTS=0short
    
    FOLDER="${FOLDER_BASE}/${TASK_NAME}/${model_name}/${NUM_SHOTS}"
    FILE=$FOLDER/metrics.json
    echo "evaluating $model_name base on $TASK_NAME $NUM_SHOTS ..."

    if [ ! -f "$FILE" ]; then
        # zero-shot
        python3 -m eval.boolq.run_eval \
            --ntrain 0 \
            --save_dir $FOLDER \
            --model_name_or_path $model_name_or_path \
            --tokenizer_name_or_path $model_name_or_path \
            --eval_batch_size 4 \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_chatml_format \
            --awq
    fi
done
