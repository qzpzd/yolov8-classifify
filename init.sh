# cd export CUDA_VISIBLE_DEVICES=0
# export PYTHONPATH="/home/disk/tanjing/ambacaffe/python":"."
#/home/disk/tanjing/anaconda3/envs/py38/bin/python /home/disk/qizhongpei/projects/ultralytics/ultralytics/yolo/v8/detect/predict.py
#/home/disk/tanjing/anaconda3/envs/py38/bin/python  /home/disk/qizhongpei/projects/ultralytics/ultralytics/yolo/v8/detect/predict.py  model=/home/disk/qizhongpei/projects/ultralytics/runs/detect/train4/weights/best.pt  source=/home/disk/qizhongpei/ssd_pytorch/OMS_phone/video/35秒右手电话未报.MP4  save_crop=True save=True
#python  /home/disk/qizhongpei/projects/ultralytics/ultralytics/yolo/v8/detect/predict.py  model=/home/disk/qizhongpei/projects/ultralytics/runs/detect/train4/weights/best.pt  source=/home/disk/qizhongpei/ssd_pytorch/OMS_phone/video/35秒右手电话未报.MP4  save_crop=True save=True
#--croped_pic false --show_video true
#/home/disk/tanjing/anaconda3/envs/py38/bin/python /home/disk/qizhongpei/ssd_pytorch/OMS_phone/phone_detection.py --model_type pytorch --show_video true --Purning true
# --Lowrank true --Binary true 
#train
python /home/disk/qizhongpei/projects/ultralytics/ultralytics/yolo/v8/classify/train.py   cfg=/home/disk/qizhongpei/projects/ultralytics/ultralytics/yolo/cfg/default.yaml
#predict
python /home/disk/qizhongpei/projects/ultralytics/ultralytics/yolo/v8/classify/predict.py   cfg=/home/disk/qizhongpei/projects/ultralytics/ultralytics/yolo/cfg/default.yaml

#export onnx
yolo export model=/home/disk/qizhongpei/projects/ultralytics/runs/classify/train10/weights/best.pt  format=onnx

#onnx-predict
更改model为onnx即可
python /home/disk/qizhongpei/projects/ultralytics/ultralytics/yolo/v8/classify/predict.py   cfg=/home/disk/qizhongpei/projects/ultralytics/ultralytics/yolo/cfg/default.yaml

