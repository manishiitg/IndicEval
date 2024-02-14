,1,2,3

MODEL_SIZE=7B
NUM_GPUS=4
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

# Lora training
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
    open_instruct/finetune.py \
    --model_name_or_path sarvamai/OpenHathi-7B-Hi-v0.1-Base \
    --use_flash_attn \
    --use_lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_target_modules q_proj v_proj down_proj gate_proj up_proj k_proj \
    --tokenizer_name sarvamai/OpenHathi-7B-Hi-v0.1-Base \
    --use_slow_tokenizer \
    --train_file data/processed/indic_instruct_data.jsonl \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --checkpointing_steps epoch \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 5e-4 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 4 \
    --output_dir output/airavata_lora/ \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1 &&

python open_instruct/merge_lora.py \
    --base_model_name_or_path sarvamai/OpenHathi-7B-Hi-v0.1-Base \
    --lora_model_name_or_path output/airavata_lora \
    --output_dir output/airavata_lora_merged
