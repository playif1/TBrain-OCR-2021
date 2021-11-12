# TBrain-OCR-2021
此專案為參加TBrain AI實戰吧 中鋼人工智慧挑戰賽-字元辨識之解題程式原始碼，大致可分為以下兩個部分。 

1. Text Detection: 偵測原始圖片的文字位置
2. Text Recognition: 辨識圖片中的文字並產出預測結果

# Text Detection
使用YOLOv5來訓練text detection model

## Installation
可參考YOLOv5 Repository的文件，基本上進入yolov5資料夾後安裝requirements.txt內的套件即可。
YOLOv5 Repository的文件: https://github.com/ultralytics/yolov5

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

# Text Recognition

