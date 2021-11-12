_base_ = ['../_base_/default_runtime.py']

checkpoint_config = dict(interval=5)
# optimizer
optimizer = dict(type='Adam', lr=1e-4)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[3, 4])
total_epochs = 20

label_convertor = dict(
    type='SegConvertor', dict_type='DICT36', with_unknown=True, lower=True)

model = dict(
    type='SegRecognizer',
    backbone=dict(
        type='ResNet31OCR',
        layers=[1, 2, 5, 3],
        channels=[32, 64, 128, 256, 512, 512],
        out_indices=[0, 1, 2, 3],
        stage4_pool_cfg=dict(kernel_size=2, stride=2),
        last_stage_pool=True),
    neck=dict(
        type='FPNOCR', in_channels=[128, 256, 512, 512], out_channels=256),
    head=dict(
        type='SegHead',
        in_channels=256,
        upsample_param=dict(scale_factor=2.0, mode='nearest')),
    loss=dict(
        type='SegLoss', seg_downsample_ratio=1.0, seg_with_loss_weight=True),
    label_convertor=label_convertor)

find_unused_parameters = True

img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

gt_label_convertor = dict(
    type='SegConvertor', dict_type='DICT36', with_unknown=True, lower=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomPaddingOCR',
        max_ratio=[0.15, 0.2, 0.15, 0.2],
        box_type='char_quads'),
    dict(type='OpencvToPil'),
    dict(
        type='RandomRotateImageBox',
        min_angle=-17,
        max_angle=17,
        box_type='char_quads'),
    dict(type='PilToOpencv'),
    dict(
        type='ResizeOCR',
        height=64,
        min_width=64,
        max_width=512,
        keep_aspect_ratio=True),
    dict(
        type='OCRSegTargets',
        label_convertor=gt_label_convertor,
        box_type='char_quads'),
    dict(type='RandomRotateTextDet', rotate_ratio=0.5, max_angle=15),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    dict(type='ToTensorOCR'),
    dict(type='FancyPCA'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(
        type='CustomFormatBundle',
        keys=['gt_kernels'],
        visualize=dict(flag=False, boundary_key=None),
        call_super=False),
    dict(
        type='Collect',
        keys=['img', 'gt_kernels'],
        meta_keys=['filename', 'ori_shape', 'resize_shape'])
]

test_img_norm_cfg = dict(
    mean=[x * 255 for x in img_norm_cfg['mean']],
    std=[x * 255 for x in img_norm_cfg['std']])
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeOCR',
        height=64,
        min_width=64,
        max_width=None,
        keep_aspect_ratio=True),
    dict(type='Normalize', **test_img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=['filename', 'ori_shape', 'resize_shape'])
]

dataset_type = 'OCRSegDataset'#'OCRSegDataset'
dataset_root = '/home/cwhuang1021/TBrain_OCR_competition/'
train_img_prefix = dataset_root + '/lmdb_train_input'
train_ann_file = dataset_root + '/lmdb_train_result/label.lmdb'

train = dict(
    type=dataset_type,
    img_prefix=train_img_prefix,
    ann_file=train_ann_file,
    loader=dict(
        type='LmdbLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator='\t')),
    pipeline=None,
    test_mode=False)

val_anno_file1 = dataset_root + '/lmdb_valid_result/label.lmdb'
val_img_prefix = dataset_root + '/lmdb_valid_input'
val = dict(
    type=dataset_type,
    img_prefix=val_img_prefix,
    ann_file=val_anno_file1,
    loader=dict(
        type='LmdbLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator='\t')),
    pipeline=None,
    test_mode=True)

dataset_type = 'OCRDataset'
test_prefix = '../CRNN_data/valid/public_validation/'
test_img_prefix = dataset_root + '/lmdb_test'
test_ann_file = dataset_root + '/lmdb_test/dummy_gt.txt'
test = dict(
    type=dataset_type,
    img_prefix=test_img_prefix,
    ann_file=test_ann_file,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator='\t')),
    pipeline=None,
    test_mode=True)



data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type='UniformConcatDataset', datasets=[train],
        pipeline=train_pipeline),
    val=dict(
        type='UniformConcatDataset',
        datasets=[val],
        pipeline=test_pipeline),
    test=dict(
        type='UniformConcatDataset',
        datasets=[test],
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='acc')
