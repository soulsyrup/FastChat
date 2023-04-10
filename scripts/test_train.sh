torchrun --nproc_per_node=4 --master_port=20001 fastchat/train/train.py \
    --model_name_or_path ~/model_weights/llama-7b  \
    --data_path ~/datasets/test.json \
    --bf16 False \
    --fp16 True \
    --output_dir output \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 False \
    --gradient_checkpointing True

    #--model_name_or_path facebook/opt-350m \
    #--data_path ~/datasets/sharegpt_backup/sharegpt_20230322_clean_lang_split.json \
    #--model_name_or_path ~/model_weights/llama-7b  \
