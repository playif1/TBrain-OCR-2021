set -e

# Launch the training with multiple GPUs
# sar: multiple GPUs training, 280s/epoch (1 GPU)
python -m torch.distributed.launch --nproc_per_node=8 --master_port=-29500 tools/train.py configs/tbrain/sar_r31_parallel_decoder_tbrain_train.py --work-dir runs/sar_r31_20211109 --launcher pytorch --no-validate

# satrn: multiple GPUs training, 240s/epoch (8 GPUs)
python -m torch.distributed.launch --nproc_per_node=8 --master_port=-29500 tools/train.py configs/tbrain/satrn_tbrain.py --work-dir runs/satrn_20211109 --launcher pytorch --no-validate

# robustscanner_r31: multiple GPUs training, 280s/epoch (1 GPU)
python -m torch.distributed.launch --nproc_per_node=8 --master_port=-29500 tools/train.py configs/tbrain/robustscanner_r31_academic.py --work-dir runs/robustscanner_r31_20211109 --launcher pytorch --no-validate

# nrtr: multiple GPUs training, 240s/epoch (1 GPU)
python -m torch.distributed.launch --nproc_per_node=8 --master_port=-29500 tools/train.py configs/tbrain/nrtr_r31_1by16_1by8_academic.py --work-dir runs/nrtr_r31_20211109 --launcher pytorch --no-validate

# Automatically remove CCS if you are using TWCC
# CCS_ID=""
# twccli rm ccs -f -s ${CCS_ID}

# Final training config below for reference.
# python tools/train.py configs/tbrain/sar_r31_parallel_decoder_tbrain_train.py --work-dir runs_private/sar1 --no-validate
# python -m torch.distributed.launch --nproc_per_node=8 --master_port=-29500 tools/train.py configs/tbrain/satrn_tbrain.py --work-dir runs_private/satrn1 --launcher pytorch --no-validate
# python tools/train.py configs/tbrain/robustscanner_r31_academic.py --work-dir runs_private/robustscanner1 --no-validate
# python tools/train.py configs/tbrain/nrtr_r31_1by16_1by8_academic.py --work-dir runs_private/nrtr1 --no-validate
