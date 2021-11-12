checkpoint_config = dict(interval=10)
log_config = dict(interval=5, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'checkpoints/sar_r31_parallel_decoder_academic-dba3a4a3.pth'
resume_from = None
workflow = [('train', 1)]
label_convertor = dict(
    type='AttnConvertor', dict_type='DICT90', with_unknown=True)
model = dict(
    type='SARNet',
    backbone=dict(type='ResNet31OCR'),
    encoder=dict(
        type='SAREncoder', enc_bi_rnn=False, enc_do_rnn=0.1, enc_gru=False),
    decoder=dict(
        type='ParallelSARDecoder',
        enc_bi_rnn=False,
        dec_bi_rnn=False,
        dec_do_rnn=0,
        dec_gru=False,
        pred_dropout=0.1,
        d_k=512,
        pred_concat=True),
    loss=dict(type='SARLoss'),
    label_convertor=dict(
        type='AttnConvertor', dict_type='DICT90', with_unknown=True),
    max_seq_len=15)
optimizer = dict(type='Adam', lr=0.001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[5, 10, 25, 40])
total_epochs = 50
img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeOCR',
        height=64,
        min_width=100,
        max_width=300,
        keep_aspect_ratio=True),
    dict(type='RandomRotateTextDet', rotate_ratio=0.5, max_angle=5),
    dict(type='ColorJitter', brightness=0.05, contrast=0.05, saturation=0.05),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'resize_shape', 'text', 'valid_ratio'
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiRotateAugOCR',
        rotate_degrees=[0, 180],
        transforms=[
            dict(
                type='ResizeOCR',
                height=64,
                min_width=100,
                max_width=300,
                keep_aspect_ratio=True),
            dict(type='ToTensorOCR'),
            dict(
                type='NormalizeOCR', mean=[0.5, 0.5, 0.5], std=[0.5, 0.5,
                                                                0.5]),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=[
                    'filename', 'ori_shape', 'resize_shape', 'valid_ratio',
                    'img_norm_cfg', 'ori_filename'
                ])
        ])
]
dataset_type = 'OCRDataset'
# TODO: Set the dataset_root for all the dataset
dataset_root = '.'
img_prefix = f'{dataset_root}/lmdb_train_input'
train_anno_file1 = f'{dataset_root}/lmdb_train_result/label.lmdb'
train1 = dict(
    type='OCRDataset',
    img_prefix=img_prefix',
    ann_file=train_anno_file1,
    loader=dict(
        type='LmdbLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator='	')),
    pipeline=None,
    test_mode=False)
img_prefix2 = f'{dataset_root}/lmdb_test'
train_anno_file2 = f'{dataset_root}/lmdb_test/pseudo_label_test.txt'
train2 = dict(
    type='OCRDataset',
    img_prefix=img_prefix2,
    ann_file=train_anno_file2,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator='	')),
    pipeline=None,
    test_mode=False)
val_anno_file1 = f'{dataset_root}/lmdb_valid_result/label.lmdb'
val_img_prefix = f'{dataset_root}/lmdb_valid_input'
val = dict(
    type='OCRDataset',
    img_prefix=val_img_prefix,
    ann_file=val_anno_file1,
    loader=dict(
        type='LmdbLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator='	')),
    pipeline=None,
    test_mode=True)

# private testing
test_anno_file1 = f'{dataset_root}/private_testing/dummy_gt.txt'
test_img_prefix = f'{dataset_root}/private_testing'
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
    workers_per_gpu=4,
    samples_per_gpu=64,
    val_dataloader=dict(samples_per_gpu=64),
    test_dataloader=dict(samples_per_gpu=256),
    train=dict(
        type='UniformConcatDataset',
        datasets=[train1, train2, val],
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
work_dir = 'runs/sar_20211108'
gpu_ids = range(0, 1)
