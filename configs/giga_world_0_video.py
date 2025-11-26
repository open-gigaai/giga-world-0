config = dict(
    runners=['giga_world_0.GigaWorld0Trainer'],
    project_dir='./experiments/giga_world_0_video/it2w',
    launch=dict(
        gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7],
        distributed_type='DEEPSPEED',
        deepspeed_config=dict(
            deepspeed_config_file='accelerate_configs/zero2.json',
        ),
    ),
    dataloaders=dict(
        train=dict(
            data_or_config=[
                '/path/to/packed_data',
            ],
            batch_size_per_gpu=1,
            num_workers=6,
            transform=dict(
                type='GigaWorld0Transform',
                num_frames=61,
                height=480,
                width=640,
                fps=16,
                image_cfg=dict(
                    mask_generator=dict(
                        max_ref_frames=1,
                        start=1,
                        factor=4,
                    ),
                ),
            ),
            sampler=dict(
                type='DefaultSampler',
                shuffle=True,
            ),
        ),
    ),
    models=dict(
        vae_model_path='/path/to/giga_world_0_video/vae',
        transformer_model_path='/path/to/giga_world_0_video/transformer',
        # train_mode='lora',
        # lora_rank=64,
    ),
    # optimizers=dict(
    #     type='AdamW',
    #     lr=2 ** (-14.5),
    #     weight_decay=1e-2,
    # ),
    optimizers=dict(
        type='CAME8Bit',
        lr=2 ** (-14.5),
    ),
    schedulers=dict(
        type='ConstantScheduler',
    ),
    train=dict(
        resume=True,
        max_epochs=1,
        gradient_accumulation_steps=1,
        mixed_precision='bf16',  # fp16, bf16, fp8
        checkpoint_interval=3,
        checkpoint_total_limit=5,
        checkpoint_strict=False,
        log_with='tensorboard',
        log_interval=1,
        with_ema=True,
        activation_checkpointing=True,
        activation_class_names=['TransformerBlock'],
    ),
)
