#!/bin/bash

source ./scripts/indic_eval/common_vars.sh

TASK_NAME=lmjudge
for model_name_or_path in "${model_names[@]}"; do
    model_name=${model_name_or_path##*/}
    
    echo "warming up $model_name base"

    if echo "$model_name" | grep -qi "awq"; then
        awq_param="--awq"
    else
        awq_param=""
    fi

    template_format="eval.templates.create_prompt_by_template"
    if echo "$model_name" | grep -qi "Airavata"; then
        template_format="eval.templates.create_prompt_with_tulu_chat_format"
    fi
    if echo "$model_name" | grep -qi "OpenHathi-7B-Hi-v0.1-Base"; then
        template_format="eval.templates.create_prompt_with_llama2_chat_format"
    fi
    if echo "$model_name" | grep -qi "OpenHermes"; then
        template_format="eval.templates.create_prompt_with_chatml_format"
    fi
    if echo "$model_name" | grep -qi "merged" && echo "$model_name" | grep -qi "gemma"; then
        template_format="eval.templates.gemma_with_chatml_format"
    fi
    
    python3 -m eval.warmup \
        --model_name_or_path $model_name_or_path \
        --tokenizer_name_or_path $model_name_or_path \
        --use_chat_format \
        --chat_formatting_function $template_format \
        $awq_param
    
done