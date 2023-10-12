python -m torch.distributed.launch --nproc_per_node 4 /leonardo/home/userexternal/elenas00/projects/huggingface_htr/src/trocr_finetune.py \
    --tokenizer processor.feature_extractor \
    --args training_args\
    --compute_metrics compute_metrics\
    --train_dataset train_concat_dataset\
    --eval_dataset test_concat_dataset\
    --data_collator default_data_collator\
    --predict_with_generate True\
    --evaluation_strategy "epoch"\
    --save_strategy "epoch"\
    --load_best_model_at_end True\
    --greater_is_better False\
    --metric_for_best_model "eval_cer"\
    --per_device_train_batch_size 32\
    --per_device_eval_batch_size 32\
    --fp16 False\
    --bf16 True\
    --output_dir "/leonardo/home/userexternal/elenas00/projects/huggingface_htr/models/trocr_test/"\
    --num_train_epochs 5\
    --save_total_limit 3\
    --warmup_steps 50\
    --weight_decay 0.01\
    --logging_steps 100\
    --save_steps 250\
    --eval_steps 250\
    --report_to 'tensorboard'\
    --seed 42