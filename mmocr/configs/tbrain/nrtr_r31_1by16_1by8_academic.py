_base_ = [
    '../_base_/default_runtime.py', '../_base_/recog_models/nrtr.py'
]

label_convertor = dict(
    type='AttnConvertor', dict_type='DICTTBRAIN', with_unknown=True, lower=False)
#     type='AttnConvertor', dict_type='DICT90', with_unknown=True)

model = dict(
    type='NRTR',
    backbone=dict(
        type='ResNet31OCR',
        layers=[1, 2, 5, 3],
        channels=[32, 64, 128, 256, 512, 512],
        stage4_pool_cfg=dict(kernel_size=(2, 1), stride=(2, 1)),
        last_stage_pool=True),
    encoder=dict(type='TFEncoder'),
    decoder=dict(type='TFDecoder'),
    loss=dict(type='TFLoss'),
    label_convertor=label_convertor,
    max_seq_len=15)
#     max_seq_len=40)

# optimizer
optimizer = dict(type='Adam', lr=1e-3)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[5, 12, 16, 30])
total_epochs = 20

img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeOCR',

        height=64, # 32
        min_width=100, # 32
        max_width=300, # 160
        keep_aspect_ratio=True),# width_downsample_ratio=0.25),
    dict(type='RandomRotateTextDet', rotate_ratio=0.5, max_angle=5),
    dict(type='ColorJitter', brightness=0.05, contrast=0.05, saturation=0.05),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'resize_shape', 'text', 'valid_ratio'
        ]),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiRotateAugOCR',
#         rotate_degrees=[0, 90, 270],        
        rotate_degrees=[0, 180],
        transforms=[
            dict(
                type='ResizeOCR',
#                 height=32,
#                 min_width=32,
#                 max_width=160,
                height=64,#32,
                min_width=100,
                max_width=300,#100,
                keep_aspect_ratio=True),#,
#                 width_downsample_ratio=0.25),
            dict(type='ToTensorOCR'),
            dict(type='NormalizeOCR', **img_norm_cfg),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=[
                    'filename', 'ori_shape', 'resize_shape', 'valid_ratio'
                ]),
        ])
]

dataset_type = 'OCRDataset'
dataset_root = '/home/cwhuang1021/TBrain_OCR_competition/'
img_prefix = dataset_root + '/lmdb_train_input'
train_anno_file1 = dataset_root + '/lmdb_train_result/label.lmdb'

train1 = dict(
    type=dataset_type,
    img_prefix=img_prefix,
    ann_file=train_anno_file1,
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

# pseudo-label dataset from public_testing
img_prefix2 = dataset_root + '/lmdb_test'
train_anno_file2 = dataset_root + '/lmdb_test/pseudo_label_test.txt'
train2 = dict(
    type=dataset_type,
    img_prefix=img_prefix2,
    ann_file=train_anno_file2,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator='\t')),
    pipeline=None,
    test_mode=False)

# pseudo-label dataset from private_testing!!!
img_prefix3 = dataset_root + '/private_testing'
train_anno_file3 = dataset_root + '/private_testing/pseudo_label_test.txt'
train3 = dict(
    type=dataset_type,
    img_prefix=img_prefix3,
    ann_file=train_anno_file3,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator='\t')),
    pipeline=None,
    test_mode=False)

# validation dataset
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

# test_anno_file1 = dataset_root + '/lmdb_test/dummy_gt.txt'
# test_img_prefix = dataset_root + '/lmdb_test'
# test = dict(
#     type=dataset_type,
#     img_prefix=test_img_prefix,
#     ann_file=test_anno_file1,
#     loader=dict(
#         type='HardDiskLoader',
#         repeat=1,
#         parser=dict(
#             type='LineStrParser',
#             keys=['filename', 'text'],
#             keys_idx=[0, 1],
#             separator='\t')),
#     pipeline=None,
#     test_mode=True)

# private testing
test_anno_file1 = dataset_root + '/private_testing/dummy_gt.txt'
test_img_prefix = dataset_root + '/private_testing'
test = dict(
    type=dataset_type,
    img_prefix=test_img_prefix,
    ann_file=test_anno_file1,
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
    samples_per_gpu=128,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=128),
    test_dataloader=dict(samples_per_gpu=128),
    train=dict(
        type='UniformConcatDataset',
        datasets=[train1, val, train2, train3],
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

# Use pretrained model
load_from = "checkpoints/nrtr_r31_academic_20210406-954db95e.pth"
