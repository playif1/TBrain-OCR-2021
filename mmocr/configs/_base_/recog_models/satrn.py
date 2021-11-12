label_convertor = dict(
    type='AttnConvertor', dict_type='DICT90', with_unknown=True)#, lower=True)

model = dict(
    type='SATRN',
    backbone=dict(type='ShallowCNN'),
    encoder=dict(type='SatrnEncoder'),
    decoder=dict(type='TFDecoder'),
    loss=dict(type='TFLoss'),
    label_convertor=label_convertor,
    max_seq_len=15)
    #max_seq_len=40)
