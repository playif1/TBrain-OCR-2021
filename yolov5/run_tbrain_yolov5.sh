# Train the text detection model, I have trained 2 models in this competition.
# TODO: Set the data_yaml based on your system. 
# Also, if GPU OOM occur, please reduce the --batch-size. If CPU OOM occur, please remove the --cache flag.
data_yaml="yolov5_data.yaml"
python train.py --weights yolov5s.pt --data ${data_yaml} --epochs 300 --batch-size 32 --imgsz 640 --workers 4 --multi-scale --project tbrain_OCR --name yolov5s --exist-ok --save-period 50 --cache
python train.py --weights yolov5m.pt --data ${data_yaml} --epochs 300 --batch-size 32 --imgsz 640 --workers 4 --multi-scale --project tbrain_OCR --name yolov5m_imgsz640_ep300 --exist-ok --save-period 50 --cache

# Perform detection on the raw image, use --save-crop to save the cropped image for mmocr.
# TODO: Change the --source to the dataset you want to detect
model_dir=""
source_dir="private_data_v2"
python detect.py --weights "${model_dir}/yolov5m_imgsz640_ep300/weights/best.pt" "${model_dir}/yolov5s/weights/best.pt" --source ${source_dir} --imgsz 640 --max-det 1 --save-txt --save-conf --save-crop --augment --project tbrain_OCR --name yolov5ms_imgsz640_ep300_final_detect

# [Optional]. Move the cropped image to another directory
detection_output_dir=""
mkdir -p ${detection_output_dir}
mv ./tbrain_OCR/yolov5ms_imgsz640_ep300_final_detect/crops/text/* ${detection_output_dir}
