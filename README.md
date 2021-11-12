# TBrain-OCR-2021
此專案為參加TBrain AI實戰吧 中鋼人工智慧挑戰賽-字元辨識之解題程式原始碼，隊伍名稱: 玖玖的奇妙冒險。

本次競賽所使用的方法大致可分為以下兩個部分，將依序說明。 

1. Text Detection: 偵測原始圖片的文字位置
2. Text Recognition: 辨識圖片中的文字並產出預測結果

---
# Text Detection
因為這次競賽的文字框大部分都是水平的矩形，因此我使用YOLOv5來訓練text detection model，以下說明安裝、資料前處理、訓練與預測的操作步驟。
* YOLOv5 Repository: https://github.com/ultralytics/yolov5

## Installation
可參考YOLOv5 Repository的文件，基本上進入yolov5資料夾後安裝requirements.txt內的套件即可。

指令如下:

```
$ cd yolov5
$ pip install -r requirements.txt
```

## Data Preprocessing
使用public_training_dataset(具有polygon標註的資料集)，開啟get_bbox_YOLOv5_label.ipynb的notebook後按照裡面的順序操作，即可產出YOLOv5格式的資料集。
接著修改yolov5/yolov5_data.yaml檔案，對應好資料集的路徑，參考如下。

```
# yolov5_data.yaml
path: ./public_yolov5_data
train: train_split/images
val: val_split/images

nc: 1
names: ['text']
```

## Model Training
參考yolov5/run_tbrain_yolov5.sh利用yolov5/train.py訓練YOLOv5s與YOLOv5m模型，如果GPU記憶體不足可降低batch size，若CPU記憶體不足則可以取消--cache參數。

```
data_yaml="yolov5_data.yaml"
python train.py --weights yolov5s.pt --data ${data_yaml} --epochs 300 --batch-size 32 --imgsz 640 --workers 4 --multi-scale --project tbrain_OCR --name yolov5s --exist-ok --save-period 50 --cache
python train.py --weights yolov5m.pt --data ${data_yaml} --epochs 300 --batch-size 32 --imgsz 640 --workers 4 --multi-scale --project tbrain_OCR --name yolov5m_imgsz640_ep300 --exist-ok --save-period 50 --cache
```

## Inference
使用訓練好的模型，將testing data具有文字的部分切出來，由於每張圖片內只有一組需要辨識的文字，因此將--max-det參數設定為1，另外設定--save-crop參數，便會直接將cropped好的圖片輸出，在Text Recognition的部分就是使用這邊產出的cropped image。

```
model_dir="."
source_dir="private_data_v2"
python detect.py --weights "${model_dir}/yolov5m_imgsz640_ep300/weights/best.pt" "${model_dir}/yolov5s/weights/best.pt" --source ${source_dir} --imgsz 640 --max-det 1 --save-txt --save-conf --save-crop --augment --project tbrain_OCR --name yolov5ms_imgsz640_ep300_final_detect
```

---
# Text Recognition
本次競賽我使用MMOCR來訓練文字辨識模型，以下說明安裝、資料前處理、訓練、預測、後處理的操作步驟。
* MMOCR: https://github.com/open-mmlab/mmocr
* MMOCR Document: https://mmocr.readthedocs.io/en/latest/

## Installation
可參考MMOCR Repository的文件，基本上進入mmocr資料夾後按照下面連結內的步驟即可。
* MMOCR Installation Guide: https://mmocr.readthedocs.io/en/latest/install.html

而對於最後的submission可分為3個round，以下一一說明。
## Round 0: 使用public dataset訓練4個基本模型
### Data Preprocessing
對於公佈的public_training_data、public_validation_data(一開始不小心釋出答案的public_testing_data)、public_testing_data_2，我使用前一章節提到的YOLOv5進行crop後，進行簡單的資料清理，將這三份資料顛倒的圖片都旋轉回來(Note: 對於private testing data我並沒有做這個人工的操作)，

### Model Training
接著使用mmocr/model_configs/round0裡的4份config檔案訓練4個不同架構的模型，指令可參考run_train_round0.sh。

1. NRTR: https://mmocr.readthedocs.io/en/latest/textrecog_models.html#nrtr
    * nrtr_r31_1by16_1by8_academic.py
2. RobustScanner: https://mmocr.readthedocs.io/en/latest/textrecog_models.html#robustscanner-dynamically-enhancing-positional-clues-for-robust-text-recognition
    * robustscanner_r31_academic.py
3. Show, Attend and Read: https://mmocr.readthedocs.io/en/latest/textrecog_models.html#show-attend-and-read-a-simple-and-strong-baseline-for-irregular-text-recognition
    * sar_r31_parallel_decoder_tbrain_train.py
4. SATRN: https://mmocr.readthedocs.io/en/latest/textrecog_models.html#satrn
    * satrn_tbrain.py
    
```
# mmocr/run_train_round0.sh
# sar: 1 GPU training, 280s/epoch
python tools/train.py model_configs/round0/sar_r31_parallel_decoder_tbrain_train.py --work-dir runs_public/sar_r31_20211109 

# satrn: multiple GPUs training, 240s/epoch (8 GPUs)
python -m torch.distributed.launch --nproc_per_node=8 --master_port=-29500 tools/train.py model_configs/round0/satrn_tbrain.py --work-dir ./runs_public/satrn_20211108 --launcher pytorch --no-validate

# robustscanner_r31: 1 GPU training, 280s/epoch
python tools/train.py model_configs/round0/robustscanner_r31_academic.py --work-dir runs_public/robustscanner_r31_20211109 

# nrtr: 1 GPU training, 240s/epoch
python tools/train.py model_configs/round0/nrtr_r31_1by16_1by8_academic.py --work-dir runs_public/nrtr_r31_20211109 
```
    
### Inference on Private Testing Dataset (Cropped by Text Detection YOLOv5 Models)
參考mmocr/run_test_round0.sh，使用訓練好的模型在private testing dataset上做預測，預測時也有使用test-time augmentation(TTA)的技巧，將圖片旋轉180度並使用信心較高的預測結果，另外利用--output-txt參數可調整輸出的預測結果之檔案名稱。

```
# mmocr/run_test_round0.sh
# satrn
python tools/test.py ./runs_public/satrn_20211108/satrn_tbrain.py ./runs_public/satrn_20211108/latest.pth --format-only --dataset-type test --output-txt sub_test_satrn.csv

# sar
python tools/test.py ./runs_public/sar_20211108/sar_r31_parallel_decoder_tbrain_train.py ./runs_public/sar_20211108/latest.pth --format-only --dataset-type test --output-txt sub_test_sar.csv

# robustscanner
python tools/test.py ./runs_public/robustscanner_r31_20211109/robustscanner_r31_academic.py ./runs_public/robustscanner_r31_20211109/latest.pth --format-only --dataset-type test --output-txt sub_test_robustscanner.csv

# nrtr
python tools/test.py ./runs_public/nrtr_r31_20211109/nrtr_r31_1by16_1by8_academic.py ./runs_public/nrtr_r31_20211109/latest.pth --format-only --dataset-type test --output-txt sub_test_nrtr.csv
```

### Generate the ensembling submission
參考get_ensemble_result.ipynb，將以上4個模型的預測結果做ensemble產出ensembling結果，此處的結果為我最後上傳的第一個submission，最終分數為106.0024。


### Get the Pseudo Label for the next round
參考get_pseudo_label.ipynb，在這個步驟中我使用SATRN的預測結果(sub_test_satrn.csv)，取出信心值最高的前8000筆資料作為pseudo labeling，加入下一輪的訓練資料。

### Automatically Recover some Ratated Image
由於顛倒的圖片對於訓練來說屬於noise(標註和圖片上的文字順序會相反)，因此我將private testing dataset與全部顛倒的private testing dataset都在沒有TTA的情況下做預測，假如顛倒後的信心分數較高則將圖片翻轉180度回來，可參考rotate_and_export.ipynb裡面的步驟，先將資料集全部顛倒，再利用model_config的設定分別做預測並讓每個模型產出兩個預測結果，再判斷顛倒與不顛倒何者信心分數較高。

## Round 1: 使用public dataset + private testing dataset (pseudo label) 重新訓練4個模型
### Model Training
接著使用mmocr/model_configs/round1裡的4份config檔案(如下)訓練4個不同架構的模型，指令可參考run_train_round1.sh。

1. nrtr_r31_1by16_1by8_academic.py
2. robustscanner_r31_academic.py
3. sar_r31_parallel_decoder_tbrain_train.py
4. satrn_tbrain.py

```
# mmocr/run_train_round1.sh
python tools/train.py model_configs/round1/sar_r31_parallel_decoder_tbrain_train.py --work-dir runs_private/sar1 --no-validate

# Very slow, use 8 gpus for training.
python -m torch.distributed.launch --nproc_per_node=8 --master_port=-29500 tools/train.py model_configs/round1/satrn_tbrain.py --work-dir runs_private/satrn1 --launcher pytorch --no-validate

python tools/train.py model_configs/round1/robustscanner_r31_academic.py --work-dir runs_private/robustscanner1 --no-validate

python tools/train.py model_configs/round1/nrtr_r31_1by16_1by8_academic.py --work-dir runs_private/nrtr1 --no-validate
```

### Inference on Private Testing Dataset (Cropped by Text Detection YOLOv5 Models)
參考mmocr/run_test_round1.sh，使用訓練好的模型在private testing dataset上做預測，其餘與round0相同。

```
# mmocr/run_test_round1.sh
# satrn
python tools/test.py ./runs_private/satrn1/satrn_tbrain.py ./runs_private/satrn1/latest.pth --format-only --dataset-type test --output-txt sub_test_satrn.csv

# sar
python tools/test.py ./runs_private/sar1/sar_r31_parallel_decoder_tbrain_train.py ./runs_private/sar1/latest.pth --format-only --dataset-type test --output-txt sub_test_sar.csv 

# robustscanner
python tools/test.py ./runs_private/robustscanner1/robustscanner_r31_academic.py ./runs_private/robustscanner1/latest.pth --format-only --dataset-type test --output-txt sub_test_robustscanner.csv 

# nrtr
python tools/test.py ./runs_private/nrtr1/nrtr_r31_1by16_1by8_academic.py ./runs_private/nrtr1/latest.pth --format-only --dataset-type test --output-txt sub_test_nrtr.csv

```

### Generate the ensembling submission
參考get_ensemble_result.ipynb，將以上4個模型的預測結果做ensemble產出ensembling結果，此階段的結果最後並未上傳。


### Get the Pseudo Label for the next round
參考get_pseudo_label.ipynb，在這個步驟中我使用ensemble後的預測結果(ensemble_4models_private_score_v1.csv)，取出信心值最高的前9000筆資料作為pseudo labeling，加入下一輪的訓練資料。

### Automatically Recover some Ratated Image
參考mmocr/run_test_round1.sh的指令，用跟round0相同的方式產生顛倒與無顛倒的預測信心分數，再自動將private testing dataset的某些圖片做旋轉。

```
# Get score for original score & reversed score  
python tools/test.py ./runs_private/satrn1/satrn_tbrain_raw.py ./runs_private/satrn1/latest.pth --format-only --dataset-type test --output-txt sub_test_satrn_raw.csv
python tools/test.py ./runs_private/satrn1/satrn_tbrain_reversed.py ./runs_private/satrn1/latest.pth --format-only --dataset-type test --output-txt sub_test_satrn_reversed.csv 

python tools/test.py ./runs_private/sar1/sar_r31_parallel_decoder_tbrain_train_raw.py ./runs_private/sar1/latest.pth --format-only --dataset-type test --output-txt sub_test_sar_raw.csv 
python tools/test.py ./runs_private/sar1/sar_r31_parallel_decoder_tbrain_train_reversed.py ./runs_private/sar1/latest.pth --format-only --dataset-type test --output-txt sub_test_sar_reversed.csv

python tools/test.py ./runs_private/robustscanner1/robustscanner_r31_academic_raw.py ./runs_private/robustscanner1/latest.pth --format-only --dataset-type test --output-txt sub_test_robustscanner_raw.csv 
python tools/test.py ./runs_private/robustscanner1/robustscanner_r31_academic_reversed.py ./runs_private/robustscanner1/latest.pth --format-only --dataset-type test --output-txt sub_test_robustscanner_reversed.csv

python tools/test.py ./runs_private/nrtr1/nrtr_r31_1by16_1by8_academic_raw.py ./runs_private/nrtr1/latest.pth --format-only --dataset-type test --output-txt sub_test_nrtr_raw.csv
python tools/test.py ./runs_private/nrtr1/nrtr_r31_1by16_1by8_academic_reversed.py ./runs_private/nrtr1/latest.pth --format-only --dataset-type test --output-txt sub_test_nrtr_reversed.csv
```

## Round 2: 使用public dataset + private testing dataset (pseudo label) 重新訓練4個模型 (與Round 1相同)
### Model Training
接著使用mmocr/model_configs/round2裡的4份config檔案(如下)訓練4個不同架構的模型，指令可參考run_train_round2.sh。

```
# mmocr/run_train_round2.sh
python tools/train.py model_configs/round2/sar_r31_parallel_decoder_tbrain_train.py --work-dir runs_private/sar2 --no-validate

# Very slow, use 8 gpus for training.
python -m torch.distributed.launch --nproc_per_node=8 --master_port=-29500 tools/train.py model_configs/round2/satrn_tbrain.py --work-dir runs_private/satrn2 --launcher pytorch --no-validate

python tools/train.py model_configs/round2/robustscanner_r31_academic.py --work-dir runs_private/robustscanner2 --no-validate

python tools/train.py model_configs/round2/nrtr_r31_1by16_1by8_academic.py --work-dir runs_private/nrtr2 --no-validate
```

### Inference on Private Testing Dataset (Cropped by Text Detection YOLOv5 Models)
參考mmocr/run_test_round2.sh，使用訓練好的模型在private testing dataset上做預測，其餘與round1相同。

```
# mmocr/run_test_round2.sh
# satrn
python tools/test.py ./runs_private/satrn2/satrn_tbrain.py ./runs_private/satrn2/latest.pth --format-only --dataset-type test --output-txt sub_test_satrn.csv

# sar
python tools/test.py ./runs_private/sar2/sar_r31_parallel_decoder_tbrain_train.py ./runs_private/sar2/latest.pth --format-only --dataset-type test --output-txt sub_test_sar.csv

# robustscanner
python tools/test.py ./runs_private/robustscanner2/robustscanner_r31_academic.py ./runs_private/robustscanner2/latest.pth --format-only --dataset-type test --output-txt sub_test_robustscanner.csv

# nrtr
python tools/test.py ./runs_private/nrtr2/nrtr_r31_1by16_1by8_academic.py ./runs_private/nrtr2/latest.pth --format-only --dataset-type test --output-txt sub_test_nrtr.csv
```

### Generate the ensembling submission
參考get_ensemble_result.ipynb，將以上4個模型的預測結果做ensemble產出ensembling結果，此處的結果為我最後上傳的第二個submission，最終分數為95.0029。
