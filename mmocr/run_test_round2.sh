set -e

############### Round 2 ###############
# satrn
python tools/test.py ./runs_private/satrn2/satrn_tbrain.py ./runs_private/satrn2/latest.pth --format-only --dataset-type test --output-txt sub_test_satrn.csv

# sar
python tools/test.py ./runs_private/sar2/sar_r31_parallel_decoder_tbrain_train.py ./runs_private/sar2/latest.pth --format-only --dataset-type test --output-txt sub_test_sar.csv

# robustscanner
python tools/test.py ./runs_private/robustscanner2/robustscanner_r31_academic.py ./runs_private/robustscanner2/latest.pth --format-only --dataset-type test --output-txt sub_test_robustscanner.csv

# nrtr
python tools/test.py ./runs_private/nrtr2/nrtr_r31_1by16_1by8_academic.py ./runs_private/nrtr2/latest.pth --format-only --dataset-type test --output-txt sub_test_nrtr.csv