set -e

python tools/train.py model_configs/round1/sar_r31_parallel_decoder_tbrain_train.py --work-dir runs_private/sar1 --no-validate

# Very slow, use 8 gpus for training.
python -m torch.distributed.launch --nproc_per_node=8 --master_port=-29500 tools/train.py model_configs/round1/satrn_tbrain.py --work-dir runs_private/satrn1 --launcher pytorch --no-validate

python tools/train.py model_configs/round1/robustscanner_r31_academic.py --work-dir runs_private/robustscanner1 --no-validate

python tools/train.py model_configs/round1/nrtr_r31_1by16_1by8_academic.py --work-dir runs_private/nrtr1 --no-validate
