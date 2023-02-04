model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet_CIFAR',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=512,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, loss_weight=1.0)))
dataset_type = 'CIFAR10'
data_preprocessor = dict(
    num_classes=10,
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    to_rgb=False)
train_pipeline = [
    dict(type='RandomCrop', crop_size=32, padding=4),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackClsInputs')
]
test_pipeline = [dict(type='PackClsInputs')]
train_dataloader = dict(
    pin_memory=True,
    persistent_workers=True,
    collate_fn=dict(type='default_collate'),
    batch_size=256,
    num_workers=2,
    dataset=dict(
        type='CIFAR10',
        data_prefix='data/cifar10',
        test_mode=False,
        pipeline=[
            dict(type='RandomCrop', crop_size=32, padding=4),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='PackClsInputs')
        ]),
    sampler=dict(type='DefaultSampler', shuffle=True))
val_dataloader = dict(
    pin_memory=True,
    persistent_workers=True,
    collate_fn=dict(type='default_collate'),
    batch_size=256,
    num_workers=2,
    dataset=dict(
        type='CIFAR10',
        data_prefix='data/cifar10/',
        test_mode=True,
        pipeline=[dict(type='PackClsInputs')]),
    sampler=dict(type='DefaultSampler', shuffle=False))
val_evaluator = dict(type='Accuracy', topk=(1, ))
test_dataloader = dict(
    pin_memory=True,
    persistent_workers=True,
    collate_fn=dict(type='default_collate'),
    batch_size=256,
    num_workers=2,
    dataset=dict(
        type='CIFAR10',
        data_prefix='data/cifar10/',
        test_mode=True,
        pipeline=[dict(type='PackClsInputs')]),
    sampler=dict(type='DefaultSampler', shuffle=False))
test_evaluator = dict(type='Accuracy', topk=(1, ))
optim_wrapper = dict(optimizer=dict(type='Adam', lr=0.1))
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[50, 75], gamma=0.1)
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=5)
val_cfg = dict()
test_cfg = dict()
auto_scale_lr = dict(base_batch_size=128)
default_scope = 'mmcls'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=20),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='VisualizationHook', enable=False))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='ClsVisualizer', vis_backends=[dict(type='LocalVisBackend')])
log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=None, deterministic=False)
launcher = 'none'
work_dir = './work_dirs/resnet18_cifar10'
