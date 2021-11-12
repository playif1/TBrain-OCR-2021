set -e

############### Round 1 ###############
# satrn
python tools/test.py ./runs_private/satrn1/satrn_tbrain.py ./runs_private/satrn1/latest.pth --format-only --dataset-type test --output-txt sub_test_satrn.csv

# sar
python tools/test.py ./runs_private/sar1/sar_r31_parallel_decoder_tbrain_train.py ./runs_private/sar1/latest.pth --format-only --dataset-type test --output-txt sub_test_sar.csv 


# robustscanner
python tools/test.py ./runs_private/robustscanner1/robustscanner_r31_academic.py ./runs_private/robustscanner1/latest.pth --format-only --dataset-type test --output-txt sub_test_robustscanner.csv 


# nrtr
python tools/test.py ./runs_private/nrtr1/nrtr_r31_1by16_1by8_academic.py ./runs_private/nrtr1/latest.pth --format-only --dataset-type test --output-txt sub_test_nrtr.csv


# Get score for original score & reversed score  
python tools/test.py ./runs_private/satrn1/satrn_tbrain_raw.py ./runs_private/satrn1/latest.pth --format-only --dataset-type test --output-txt sub_test_satrn_raw.csv
python tools/test.py ./runs_private/satrn1/satrn_tbrain_reversed.py ./runs_private/satrn1/latest.pth --format-only --dataset-type test --output-txt sub_test_satrn_reversed.csv 

python tools/test.py ./runs_private/sar1/sar_r31_parallel_decoder_tbrain_train_raw.py ./runs_private/sar1/latest.pth --format-only --dataset-type test --output-txt sub_test_sar_raw.csv 
python tools/test.py ./runs_private/sar1/sar_r31_parallel_decoder_tbrain_train_reversed.py ./runs_private/sar1/latest.pth --format-only --dataset-type test --output-txt sub_test_sar_reversed.csv

python tools/test.py ./runs_private/robustscanner1/robustscanner_r31_academic_raw.py ./runs_private/robustscanner1/latest.pth --format-only --dataset-type test --output-txt sub_test_robustscanner_raw.csv 
python tools/test.py ./runs_private/robustscanner1/robustscanner_r31_academic_reversed.py ./runs_private/robustscanner1/latest.pth --format-only --dataset-type test --output-txt sub_test_robustscanner_reversed.csv

python tools/test.py ./runs_private/nrtr1/nrtr_r31_1by16_1by8_academic_raw.py ./runs_private/nrtr1/latest.pth --format-only --dataset-type test --output-txt sub_test_nrtr_raw.csv
python tools/test.py ./runs_private/nrtr1/nrtr_r31_1by16_1by8_academic_reversed.py ./runs_private/nrtr1/latest.pth --format-only --dataset-type test --output-txt sub_test_nrtr_reversed.csv
