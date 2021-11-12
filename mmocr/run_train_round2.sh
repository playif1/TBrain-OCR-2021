set -e

python tools/train.py model_configs/round2/sar_r31_parallel_decoder_tbrain_train.py --work-dir runs_private/sar2 --no-validate

# Very slow, use 8 gpus for training.
python -m torch.distributed.launch --nproc_per_node=8 --master_port=-29500 tools/train.py model_configs/round2/satrn_tbrain.py --work-dir runs_private/satrn2 --launcher pytorch --no-validate

python tools/train.py model_configs/round2/robustscanner_r31_academic.py --work-dir runs_private/robustscanner2 --no-validate

python tools/train.py model_configs/round2/nrtr_r31_1by16_1by8_academic.py --work-dir runs_private/nrtr2 --no-validate
