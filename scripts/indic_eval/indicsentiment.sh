export CUDA_VISIBLE_DEVICES=0



model_names=(
    "manishiitg/open-aditi-hi-v2-awq"
    "manishiitg/open-aditi-hi-v1-awq"
    "TheBloke/OpenHermes-2.5-Mistral-7B-AWQ"
)
FOLDER_BASE=/sky-notebook/eval-results


for model_name_or_path in "${model_names[@]}"; do
    model_name=${model_name_or_path##*/}
    TASK_NAME=indicsentiment
    NUM_SHOTS=0short
    
    FOLDER="${FOLDER_BASE}/${TASK_NAME}/${model_name}/${NUM_SHOTS}"
    FILE=$FOLDER/metrics.json

    if [ ! -f "$FILE" ]; then
        # zero-shot
        python3 -m eval.indicsentiment.run_translate_test_eval_exact \
            --ntrain 0 \
            --save_dir $FOLDER \
            --model_name_or_path $model_name_or_path \
            --tokenizer_name_or_path $model_name_or_path \
            --eval_batch_size 8 \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_chatml_format \
            --use_vllm \
            --awq
    else
        cat "$FILE"
    fi
done
