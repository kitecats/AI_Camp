default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend'),
                  dict(type='WandbVisBackend')],
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=100,
    val_interval=10,
    dynamic_intervals=[(90, 1)])
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-05, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=2.5e-05,
        begin=50,
        end=100,
        T_max=50,
        by_epoch=True,
        convert_to_iter_based=True)
]
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=0.008, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0,
        bias_decay_mult=0.0,
        bypass_duplicate=True,
        bias_lr_mult=2.0),
    clip_grad=dict(max_norm=35, norm_type=2),
    loss_scale='dynamic')
auto_scale_lr = dict(enable=False, base_batch_size=16)
dataset_type = 'CocoDataset'
data_root = './data/coco/'
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CachedMosaic', img_scale=(640, 640), pad_val=114.0),
    dict(
        type='RandomResize',
        scale=(1280, 1280),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(
        type='CachedMixUp',
        img_scale=(640, 640),
        ratio_range=(1.0, 1.0),
        max_cached_images=20,
        pad_val=(114, 114, 114)),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=32,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type='CocoDataset',
        data_root='./data/coco/',
        ann_file='annotations/voc12_train.json',
        data_prefix=dict(img='images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='CachedMosaic', img_scale=(640, 640), pad_val=114.0),
            dict(
                type='RandomResize',
                scale=(1280, 1280),
                ratio_range=(0.5, 2.0),
                keep_ratio=True),
            dict(type='RandomCrop', crop_size=(640, 640)),
            dict(type='YOLOXHSVRandomAug'),
            dict(type='RandomFlip', prob=0.5),
            dict(
                type='Pad', size=(640, 640),
                pad_val=dict(img=(114, 114, 114))),
            dict(
                type='CachedMixUp',
                img_scale=(640, 640),
                ratio_range=(1.0, 1.0),
                max_cached_images=20,
                pad_val=(114, 114, 114)),
            dict(type='PackDetInputs')
        ],
        metainfo=dict(
            classes=('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                     'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
                     'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                     'sofa', 'train', 'tvmonitor'),
            palette=[(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                     (197, 226, 255),
                     (0, 60, 100), (0, 0, 142), (255, 77, 255), (153, 69, 1),
                     (120, 166, 157), (0, 182, 199), (0, 226, 252),
                     (182, 182, 255), (0, 0, 230), (220, 20, 60),
                     (163, 255, 0), (0, 82, 0), (3, 95, 161), (0, 80, 100),
                     (183, 130, 88)])),
    pin_memory=True)
val_dataloader = dict(
    batch_size=5,
    num_workers=10,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root='./data/coco/',
        ann_file='annotations/voc12_val.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='Resize', scale=(640, 640), keep_ratio=True),
            dict(
                type='Pad', size=(640, 640),
                pad_val=dict(img=(114, 114, 114))),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ],
        metainfo=dict(
            classes=('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                     'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
                     'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                     'sofa', 'train', 'tvmonitor'),
            palette=[(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                     (197, 226, 255),
                     (0, 60, 100), (0, 0, 142), (255, 77, 255), (153, 69, 1),
                     (120, 166, 157), (0, 182, 199), (0, 226, 252),
                     (182, 182, 255), (0, 0, 230), (220, 20, 60),
                     (163, 255, 0), (0, 82, 0), (3, 95, 161), (0, 80, 100),
                     (183, 130, 88)])))
test_dataloader = dict(
    batch_size=5,
    num_workers=10,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root='./data/coco/',
        ann_file='annotations/voc12_val.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='Resize', scale=(640, 640), keep_ratio=True),
            dict(
                type='Pad', size=(640, 640),
                pad_val=dict(img=(114, 114, 114))),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ],
        metainfo=dict(
            classes=('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                     'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
                     'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                     'sofa', 'train', 'tvmonitor'),
            palette=[(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                     (197, 226, 255),
                     (0, 60, 100), (0, 0, 142), (255, 77, 255), (153, 69, 1),
                     (120, 166, 157), (0, 182, 199), (0, 226, 252),
                     (182, 182, 255), (0, 0, 230), (220, 20, 60),
                     (163, 255, 0), (0, 82, 0), (3, 95, 161), (0, 80, 100),
                     (183, 130, 88)])))
val_evaluator = dict(
    type='CocoMetric',
    ann_file='./data/coco/annotations/voc12_val.json',
    metric='bbox',
    format_only=False,
    proposal_nums=(100, 1, 10))
test_evaluator = dict(
    type='CocoMetric',
    ann_file='data/coco/annotations/instances_val2017.json',
    metric='bbox',
    format_only=False,
    proposal_nums=(100, 1, 10))
model = dict(
    type='RTMDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        batch_augments=None),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.33,
        widen_factor=0.5,
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint=
            'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e.pth'
        )),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='RTMDetSepBNHead',
        num_classes=20,
        in_channels=128,
        stacked_convs=2,
        feat_channels=128,
        anchor_generator=dict(
            type='MlvlPointGenerator', offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        with_objectness=False,
        exp_on_reg=False,
        share_conv=True,
        pred_kernel_size=1,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    train_cfg=dict(
        assigner=dict(type='DynamicSoftLabelAssigner', topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=30000,
        min_bbox_size=0,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=300))
train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=(640, 640),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]
max_epochs = 100
stage2_num_epochs = 10
base_lr = 0.0005
interval = 10
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=90,
        switch_pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='RandomResize',
                scale=(640, 640),
                ratio_range=(0.5, 2.0),
                keep_ratio=True),
            dict(type='RandomCrop', crop_size=(640, 640)),
            dict(type='YOLOXHSVRandomAug'),
            dict(type='RandomFlip', prob=0.5),
            dict(
                type='Pad', size=(640, 640),
                pad_val=dict(img=(114, 114, 114))),
            dict(type='PackDetInputs')
        ])
]
checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e.pth'
METAINFO = dict(
    classes=('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
             'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
             'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'),
    palette=[(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
             (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
             (153, 69, 1), (120, 166, 157), (0, 182, 199), (0, 226, 252),
             (182, 182, 255), (0, 0, 230), (220, 20, 60), (163, 255, 0),
             (0, 82, 0), (3, 95, 161), (0, 80, 100), (183, 130, 88)])
launcher = 'none'
work_dir = './work_dirs/rtmdet_voc'