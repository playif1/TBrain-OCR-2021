set -e

# Perform inference on test dataset
# python tools/test.py ${config_path} ${weight_path} --format-only --dataset-type test

############### Round 0 ###############
# satrn
python tools/test.py ./runs_public/satrn_20211108/satrn_tbrain.py ./runs_public/satrn_20211108/latest.pth --format-only --dataset-type test --output-txt sub_test_satrn.csv

# sar
python tools/test.py ./runs_public/sar_20211108/sar_r31_parallel_decoder_tbrain_train.py ./runs_public/sar_20211108/latest.pth --format-only --dataset-type test --output-txt sub_test_sar.csv

# robustscanner
python tools/test.py ./runs_public/robustscanner_r31_20211109/robustscanner_r31_academic.py ./runs_public/robustscanner_r31_20211109/latest.pth --format-only --dataset-type test --output-txt sub_test_robustscanner.csv

# nrtr
python tools/test.py ./runs_public/nrtr_r31_20211109/nrtr_r31_1by16_1by8_academic.py ./runs_public/nrtr_r31_20211109/latest.pth --format-only --dataset-type test --output-txt sub_test_nrtr.csv
