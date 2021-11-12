_base_ = [
    '../_base_/default_runtime.py', '../_base_/recog_models/satrn.py'
]

label_convertor = dict(
    type='AttnConvertor', dict_type='DICTTBRAIN', with_unknown=True)
    #type='AttnConvertor', dict_type='DICT90', with_unknown=True)

model = dict(
    type='SATRN',
    backbone=dict(type='ShallowCNN', input_channels=3, hidden_dim=512),
    encoder=dict(
        type='SatrnEncoder',
        n_layers=12,
        n_head=8,
        d_k=512 // 8,
        d_v=512 // 8,
        d_model=512,
        n_position=100,
        d_inner=512 * 4,
        dropout=0.1),
    decoder=dict(
        type='TFDecoder',
        n_layers=6,
        d_embedding=512,
        n_head=8,
        d_model=512,
        d_inner=512 * 4,
        d_k=512 // 8,
        d_v=512 // 8),
    loss=dict(type='TFLoss'),
    label_convertor=label_convertor,
    max_seq_len=15)
    #max_seq_len=25)

# optimizer
optimizer = dict(type='Adam', lr=3e-4) # dict(type='Adam', lr=1e-6)#
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[5, 12, 16, 30])
total_epochs = 20

img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeOCR',
        height=64,#32,
        min_width=100,
        max_width=300,#100,
        keep_aspect_ratio=True),
        #keep_aspect_ratio=False,
        #width_downsample_ratio=0.25),
    dict(type='RandomRotateTextDet', rotate_ratio=0.5, max_angle=5),
    dict(type='ColorJitter', brightness=0.05, contrast=0.05, saturation=0.05),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'img_shape', 'text', 'valid_ratio',
            'resize_shape'
        ]),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiRotateAugOCR',
        #rotate_degrees=[0, 90, 270],
        rotate_degrees=[0, 180],
        transforms=[
            dict(
                type='ResizeOCR',
                height=64,#32,
                min_width=100,
                max_width=300,#100,
                keep_aspect_ratio=True),
                #keep_aspect_ratio=False,
                #width_downsample_ratio=0.25),
            dict(type='ToTensorOCR'),
            dict(type='NormalizeOCR', **img_norm_cfg),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=[
                    'filename', 'ori_shape', 'img_shape', 'valid_ratio',
                    'resize_shape'
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
    samples_per_gpu=8,#64,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=8),
    test_dataloader=dict(samples_per_gpu=64),
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
load_from = 'checkpoints/satrn_academic_20211009-cb8b1580.pth'