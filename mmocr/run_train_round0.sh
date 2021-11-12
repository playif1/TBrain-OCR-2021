set -e

python tools/train.py configs/tbrain/satrn_tbrain.py --work-dir runs/satrn_20211105

# sar: 1 GPU training, 280s/epoch
python tools/train.py model_configs/round0/sar_r31_parallel_decoder_tbrain_train.py --work-dir runs_public/sar_r31_20211109 

# satrn: multiple GPUs training, 240s/epoch (8 GPUs)
python -m torch.distributed.launch --nproc_per_node=8 --master_port=-29500 tools/train.py model_configs/round0/satrn_tbrain.py --work-dir ./runs_public/satrn_20211108 --launcher pytorch --no-validate

# robustscanner_r31: 1 GPU training, 280s/epoch
python tools/train.py model_configs/round0/robustscanner_r31_academic.py --work-dir runs_public/robustscanner_r31_20211109 

# nrtr: 1 GPU training, 240s/epoch
python tools/train.py model_configs/round0/nrtr_r31_1by16_1by8_academic.py --work-dir runs_public/nrtr_r31_20211109 
