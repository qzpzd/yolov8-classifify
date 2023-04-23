# Ultralytics YOLO 🚀, GPL-3.0 license

import glob
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
import torch
from PIL import Image

from ultralytics.yolo.data.augment import LetterBox
from ultralytics.yolo.data.utils import IMG_FORMATS, VID_FORMATS
from ultralytics.yolo.utils import LOGGER, ROOT, is_colab, is_kaggle, ops
from ultralytics.yolo.utils.checks import check_requirements


# #-----------------landmarks_detect-------------------------------------------
# import caffe
# model_def = "/home/disk/tanjing/qizhognpei/ultralytics/ultralytics/yolo/engine/prnet128_256_focus_defocus_4d_stream.prototxt"
# model_weights = "/home/disk/tanjing/qizhognpei/ultralytics/ultralytics/yolo/engine/prnet128_exp154.caffemodel"
# boxnet = caffe.Net(model_def,model_weights,caffe.TEST)
# #---------------------img_preprocess-----------------------------------------------------------------
# def img_preprocess(frame,imgsz):
#     img = cv2.resize(frame,(imgsz,imgsz))
#     img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     img_gray = img_gray[:,:,np.newaxis]
#     #img = np.concatenate((img_gray, img_gray, img_gray), axis=-1)
#     img = img_gray[:, :, ::-1].transpose(2, 0, 1).astype('float32')[np.newaxis, :, :, :]
#     img /= 255.0  
#     return img
# #--------------------crop_calling_area-------------------------------------------------------------------
# def bbox_from_points(points):#获取landmark外接矩形框
#     max_=np.max(points,axis=0)
#     min_=np.min(points,axis=0)
#     return [min_[0],min_[1],max_[0],max_[1]]

# def crop_calling_area(lmks,h,w):
#     eye_index = [41,37,28,32]
#     center = np.mean(lmks[eye_index,:],axis = 0) 
#     face_bbox=bbox_from_points(lmks)
#     bbox_w = face_bbox[2]-face_bbox[0]
#     bbox_h = face_bbox[3]-face_bbox[1]

#     # x0 =  np.clip(face_bbox[0] - 1*bbox_w,0,w-1)
#     # y0 =  np.clip(face_bbox[1] - 0.5*bbox_h,0,h-1)
#     # x1 =  np.clip(face_bbox[2] + 1*bbox_w,0,w)
#     # y1=  np.clip(face_bbox[3] + 0.5*bbox_h,0,h)
#     #print(face_bbox[0],face_bbox[1],face_bbox[2],face_bbox[3])

#     x0 =  np.clip(face_bbox[0] - 1.2*bbox_w,0,w-1)
#     y0 =  np.clip(face_bbox[1] - 1.2*bbox_h,0,h-1)
#     x1 =  np.clip(face_bbox[2] + 1*bbox_w,0,w)
#     y1=  np.clip(face_bbox[3] + 1*bbox_h,0,h)
#     return [x0,y0,x1,y1]
# #-----------------------------------------face_detect---------------------------
# import onnxruntime
# import numpy as np

# face_model = "/home/disk/tanjing/qizhognpei/ultralytics/ultralytics/yolo/engine/exp30.onnx"
# session = onnxruntime.InferenceSession(face_model)
# in_name = [input.name for input in session.get_inputs()][0]
# out_name = [output.name for output in session.get_outputs()]

# #imgsz = 480
# anchors = np.fromfile('/home/disk/tanjing/qizhognpei/ultralytics/ultralytics/yolo/engine/face_480_priorbox_220527.txt',sep=' ')
# anchors = anchors.reshape(-1,5)
# def xywh2xyxy(x):
#     # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
#     y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
#     y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
#     y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
#     y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
#     y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
#     return y
# def py_cpu_nms(dets0, conf_thresh, iou_thresh):
#     """Pure Python NMS baseline."""
#     nc = dets0.shape[1] - 5
#     dets = dets0[dets0[:, 4] > conf_thresh]
#     dets = xywh2xyxy(dets)
    
#     keep_all = []
#     for cls in range(nc):
#         dets_single = dets[np.argmax(dets[:,5:],axis=1)==cls]
#         #print('dets_single %d'%cls,dets_single)
#         x1 = dets_single[:, 0]
#         y1 = dets_single[:, 1]
#         x2 = dets_single[:, 2]
#         y2 = dets_single[:, 3]
#         scores = dets_single[:, 4]
#         areas = (x2 - x1 + 1) * (y2 - y1 + 1) 
#         order = scores.argsort()[::-1]
#         keep = []
#         while order.size > 0:
#             i = order[0]
#             keep.append(i)  
#             xx1 = np.maximum(x1[i], x1[order[1:]])
#             yy1 = np.maximum(y1[i], y1[order[1:]])
#             xx2 = np.minimum(x2[i], x2[order[1:]])
#             yy2 = np.minimum(y2[i], y2[order[1:]])
#             w = np.maximum(0.0, xx2 - xx1 + 1)
#             h = np.maximum(0.0, yy2 - yy1 + 1)
#             inter = w * h
#             ovr = inter / (areas[i] + areas[order[1:]] - inter)
#             inds = np.where(ovr <= iou_thresh)[0]
#             order = order[inds + 1]
#         keep_rect = dets_single[keep]
#         #print('keep',keep)
#         keep_all.extend(keep_rect)
#     return keep_all


# def np_sigmoid(x):
#     return 1.0/(1.0+1.0/np.exp(x))
# def decode_output(pred_raw_data,anchor_txt):
#     pred_raw_data = np_sigmoid(pred_raw_data)
#     pred_raw_data[:, 0] = (pred_raw_data[:, 0] * 2. - 0.5 + anchor_txt[:, 0]) * anchor_txt[:, 4] #x
#     pred_raw_data[:, 1] = (pred_raw_data[:, 1] * 2. - 0.5 + anchor_txt[:, 1]) * anchor_txt[:, 4] #y
#     pred_raw_data[:, 2] = (pred_raw_data[:, 2] * 2) ** 2 * anchor_txt[:, 2]  # w
#     pred_raw_data[:, 3] = (pred_raw_data[:, 3] * 2) ** 2 * anchor_txt[:, 3]  # h

#     return pred_raw_data
# def face_detect(img):
#     pred = session.run(out_name,{in_name: img})
#     x1 = np.array(pred[0]).reshape(-1, 6)
#     x2 = np.array(pred[1]).reshape(-1, 6)
#     x3 = np.array(pred[2]).reshape(-1, 6)
#     out_data_raw = np.vstack((x1,x2,x3))
#     output_from_txt = decode_output(out_data_raw,anchors)
#     pred = py_cpu_nms(output_from_txt, 0.5, 0.45)
#     return pred
# def scale_ratio(each,frame,imgsz):
#     ratio = (frame.shape[0] /imgsz , frame.shape[1] / imgsz)
#     each[[0, 2]] *= ratio[1]
#     each[[1, 3]] *= ratio[0]
#     return each
# def bbox_enlarger(img_width,img_height,bbox,scale):
#     w=bbox[2]-bbox[0]+1
#     h=bbox[3]-bbox[1]+1
#     bbox0=np.clip(bbox[0]-w*scale[0],0,img_width)
#     bbox1=np.clip(bbox[1]-h*scale[1],0,img_height)
#     bbox2=np.clip(bbox[2]+w*scale[0],0,img_width)
#     bbox3=np.clip(bbox[3]+0.8*h*scale[1],0,img_height)
#     return np.array([bbox0,bbox1,bbox2,bbox3])
# def get_lmk(bbox,img):
#     points = []
#     face_crop = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
#     point3d = get_point3d(boxnet,face_crop)
#     point3d = point3d + np.array([int(bbox[0]),int(bbox[1]),0]).reshape(-1,3)
#     for i,point2d in enumerate(point3d[:,0:2]):
#         point2d = point2d.astype(int)
#         points.append([point2d[0],point2d[1]])
#     points = np.array(points)
#     return points

# def get_point3d(net,image):
#     height,width,channel = image.shape
#     image = cv2.resize(image,(128,128))  
#     if image.ndim == 3 and channel == 3:
#         image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#     if image.ndim==2:
#         image=image[:,:,np.newaxis]
#     net_input=(torch.from_numpy(np.transpose(image,(2,0,1))).unsqueeze(0).cuda()-128.0)/128.0   
#     face_input_numpy = net_input.cpu().float().numpy().copy()
#     net.blobs['data'].data[:] = face_input_numpy.copy()[:].astype('float64')
#     net.forward()
#     caffe_pred=net.blobs['interp10'].data.copy()[0]
#     pos_map=np.transpose(caffe_pred,(1,2,0))

#     uv_kpt_ind = np.fromfile('/home/disk/tanjing/qizhognpei/ultralytics/ultralytics/yolo/engine/uv_kpt_ind_lm67.txt',sep=' ').reshape(2,-1).astype(int)
#     lmk67_temp = pos_map[uv_kpt_ind[1,:], uv_kpt_ind[0,:], :]
#     lmk67_3d=(lmk67_temp*np.array([width,height,1]).reshape(-1,3)).astype(int)

#     return lmk67_3d

# -*- coding: utf-8 -*
import argparse
import os
import cv2
import torch
import sys
sys.path.append("/home/disk/qizhongpei/projects/bodyhands/")

from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.config import get_cfg
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
import sys
sys.path.append("/home/disk/qizhongpei/projects/bodyhands/BodyHands/")
from bodyhands import *
from datasets import *
from bodyhands import add_bodyhands_config
from bodyhands import CustomVisualizer

class CustomPredictor:

    def __init__(self, cfg):
        self.cfg = cfg.clone()  
        self.model = build_model(self.cfg)
        self.model.eval()
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        with torch.no_grad():  
            if self.input_format == "RGB":
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = torch.as_tensor(original_image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs], height, width)[0]
            return predictions

def prepareModel(cfg_file, weights, thresh):
    cfg = get_cfg()
    add_bodyhands_config(cfg)
    cfg.merge_from_file(cfg_file)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh
    cfg.MODEL.WEIGHTS = os.path.abspath(weights)
    predictor = CustomPredictor(cfg)
    return predictor









@dataclass
class SourceTypes:
    webcam: bool = False
    screenshot: bool = False
    from_img: bool = False


class LoadStreams:
    # YOLOv8 streamloader, i.e. `yolo predict source='rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP streams`
    def __init__(self, sources='file.streams', imgsz=640, stride=32, auto=True, transforms=None, vid_stride=1):
        torch.backends.cudnn.benchmark = True  # faster for fixed-size inference
        self.mode = 'stream'
        self.imgsz = imgsz
        self.stride = stride
        self.vid_stride = vid_stride  # video frame-rate stride
        sources = Path(sources).read_text().rsplit() if os.path.isfile(sources) else [sources]
        n = len(sources)
        self.sources = [ops.clean_str(x) for x in sources]  # clean source names for later
        self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n
        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            st = f'{i + 1}/{n}: {s}... '
            if urlparse(s).hostname in ('www.youtube.com', 'youtube.com', 'youtu.be'):  # if source is YouTube video
                # YouTube format i.e. 'https://www.youtube.com/watch?v=Zgi9g1ksQHc' or 'https://youtu.be/Zgi9g1ksQHc'
                check_requirements(('pafy', 'youtube_dl==2020.12.2'))
                import pafy  # noqa
                s = pafy.new(s).getbest(preftype="mp4").url  # YouTube URL
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            if s == 0 and (is_colab() or is_kaggle()):
                raise NotImplementedError("'source=0' webcam not supported in Colab and Kaggle notebooks. "
                                          "Try running 'source=0' in a local environment.")
            cap = cv2.VideoCapture(s)
            if not cap.isOpened():
                raise ConnectionError(f'{st}Failed to open {s}')
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)  # warning: may return 0 or nan
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback
            self.fps[i] = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback

            success, self.imgs[i] = cap.read()  # guarantee first frame
            if not success or self.imgs[i] is None:
                raise ConnectionError(f'{st}Failed to read images from {s}')
            self.threads[i] = Thread(target=self.update, args=([i, cap, s]), daemon=True)
            LOGGER.info(f"{st}Success ✅ ({self.frames[i]} frames of shape {w}x{h} at {self.fps[i]:.2f} FPS)")
            self.threads[i].start()
        LOGGER.info('')  # newline

        # check for common shapes
        s = np.stack([LetterBox(imgsz, auto, stride=stride)(image=x).shape for x in self.imgs])
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        self.auto = auto and self.rect
        self.transforms = transforms  # optional
        self.bs = self.__len__()

        if not self.rect:
            LOGGER.warning('WARNING ⚠️ Stream shapes differ. For optimal performance supply similarly-shaped streams.')

    def update(self, i, cap, stream):
        # Read stream `i` frames in daemon thread
        n, f = 0, self.frames[i]  # frame number, frame array
        while cap.isOpened() and n < f:
            n += 1
            cap.grab()  # .read() = .grab() followed by .retrieve()
            if n % self.vid_stride == 0:
                success, im = cap.retrieve()
                if success:
                    self.imgs[i] = im
                else:
                    LOGGER.warning('WARNING ⚠️ Video stream unresponsive, please check your IP camera connection.')
                    self.imgs[i] = np.zeros_like(self.imgs[i])
                    cap.open(stream)  # re-open stream if signal was lost
            time.sleep(0.0)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        im0 = self.imgs.copy()
        if self.transforms:
            im = np.stack([self.transforms(x) for x in im0])  # transforms
        else:
            im = np.stack([LetterBox(self.imgsz, self.auto, stride=self.stride)(image=x) for x in im0])
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
            im = np.ascontiguousarray(im)  # contiguous

        return self.sources, im, im0, None, ''

    def __len__(self):
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years


class LoadScreenshots:
    # YOLOv8 screenshot dataloader, i.e. `yolo predict source=screen`
    def __init__(self, source, imgsz=640, stride=32, auto=True, transforms=None):
        # source = [screen_number left top width height] (pixels)
        check_requirements('mss')
        import mss  # noqa

        source, *params = source.split()
        self.screen, left, top, width, height = 0, None, None, None, None  # default to full screen 0
        if len(params) == 1:
            self.screen = int(params[0])
        elif len(params) == 4:
            left, top, width, height = (int(x) for x in params)
        elif len(params) == 5:
            self.screen, left, top, width, height = (int(x) for x in params)
        self.imgsz = imgsz
        self.stride = stride
        self.transforms = transforms
        self.auto = auto
        self.mode = 'stream'
        self.frame = 0
        self.sct = mss.mss()
        self.bs = 1

        # Parse monitor shape
        monitor = self.sct.monitors[self.screen]
        self.top = monitor["top"] if top is None else (monitor["top"] + top)
        self.left = monitor["left"] if left is None else (monitor["left"] + left)
        self.width = width or monitor["width"]
        self.height = height or monitor["height"]
        self.monitor = {"left": self.left, "top": self.top, "width": self.width, "height": self.height}

    def __iter__(self):
        return self

    def __next__(self):
        # mss screen capture: get raw pixels from the screen as np array
        im0 = np.array(self.sct.grab(self.monitor))[:, :, :3]  # [:, :, :3] BGRA to BGR
        s = f"screen {self.screen} (LTWH): {self.left},{self.top},{self.width},{self.height}: "

        if self.transforms:
            im = self.transforms(im0)  # transforms
        else:
            im = LetterBox(self.imgsz, self.auto, stride=self.stride)(image=im0)
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous
        self.frame += 1
        return str(self.screen), im, im0, None, s  # screen, img, original img, im0s, s


class LoadImages:
    # YOLOv8 image/video dataloader, i.e. `yolo predict source=image.jpg/vid.mp4`
    def __init__(self, path, imgsz=640, stride=32, auto=True, transforms=None, vid_stride=1):
        if isinstance(path, str) and Path(path).suffix == ".txt":  # *.txt file with img/vid/dir on each line
            path = Path(path).read_text().rsplit()
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            p = str(Path(p).resolve())
            if '*' in p:
                files.extend(sorted(glob.glob(p, recursive=True)))  # glob
            elif os.path.isdir(p):
                files.extend(sorted(glob.glob(os.path.join(p, '*.*'))))  # dir
            elif os.path.isfile(p):
                files.append(p)  # files
            else:
                raise FileNotFoundError(f'{p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.imgsz = imgsz
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.auto = auto
        self.transforms = transforms  # optional
        self.vid_stride = vid_stride  # video frame-rate stride
        self.bs = 1
        if any(videos):
            self.orientation = None  # rotation degrees
            self._new_video(videos[0])  # new video
        else:
            self.cap = None
        if self.nf == 0:
            raise FileNotFoundError(f'No images or videos found in {p}. '
                                    f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}')

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            for _ in range(self.vid_stride):
                self.cap.grab()
            success, im0 = self.cap.retrieve()
            while not success:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                path = self.files[self.count]
                self._new_video(path)
                success, im0 = self.cap.read()
            
            #---------------------------添加人脸框------------------------------------
            # im0_ = im0
            # img = img_preprocess(im0,480)
            # pred = face_detect(img)   
            # height,width,ch=im0.shape    
            # #for each in pred:
            # #print(pred[0])
            # each = scale_ratio(pred[0],im0,480)
            # area=bbox_enlarger(im0.shape[1],im0.shape[0],each,scale=[0.7,0.5]).astype(int)
            # #print(bbox[0],bbox[2],bbox[1],bbox[3])
            # # if bbox[0]==bbox[2] or bbox[1]==bbox[3]:
            # #     continue
            # # lmks = get_lmk(bbox,im0)
            # # area = crop_calling_area(lmks,height,width)
            # #print(int(area[1]),int(area[3]),int(area[0]),(area[2]))
            # im0 = im0[int(area[1]):int(area[3]),int(area[0]):int(area[2])]
            # im0 = np.ascontiguousarray(im0)
            # cv2.rectangle(im0_,(int(area[0]),int(area[1])),(int(area[2]),int(area[3])),(0,255,255),2)
            #--------------------------------------------------------------

            
            self.frame += 1
            # im0 = self._cv2_rotate(im0)  # for use if cv2 autorotation is False
            s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '

        else:
            # Read image
            self.count += 1
            im0 = cv2.imread(path)  # BGR
            if im0 is None:
                raise FileNotFoundError(f'Image Not Found {path}')
            s = f'image {self.count}/{self.nf} {path}: '


        #--------------------------------hand_detection-----------------------
        while True:
            roi_score_thresh = float(0.7)
            model = prepareModel('/home/disk/qizhongpei/projects/bodyhands/BodyHands/configs/BodyHands.yaml', '/home/disk/qizhongpei/projects/bodyhands/BodyHands/models/model.pth', roi_score_thresh)

            im0_ = im0
            height, width = im0.shape[:2]
            #print(height, width)
            ratio = width / height
            outheight = 720
            outwidth = int(ratio * outheight)
            im0_ = cv2.resize(im0_, (outwidth, outheight))
            outputs = model(im0_)
            v = CustomVisualizer(im0_[:, :, ::-1], MetadataCatalog.get("HandBodyContactHands_sub"), scale=1.0)
            #######################################################################################
            outputs = outputs["instances"].to("cpu")
            classes = outputs.pred_classes
            body_ids = outputs.pred_body_ids
            boxes = outputs.pred_boxes.tensor
            masks = outputs.pred_masks
            hand_indices = classes == 0
            body_indices = classes == 1
            hand_boxes = boxes[hand_indices]

            ratio_w = float(width/outwidth*1.0)
            ratio_h = float(height/outheight*1.0)
            #ratio1 = height / 720

            import numpy as np
            
            if not hand_boxes.numel():
            
                print("没有检测到手部")
                
                im_ = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)#添加单通道

                if self.transforms:
                    im = self.transforms(im_)  # transforms
                else:
                    im = LetterBox(self.imgsz, self.auto, stride=self.stride)(image=im_)
                    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                    im = np.ascontiguousarray(im)  # contiguous
                
                area = [0, 0, width, height]#添加区域用于书写文本
                return path, im, im0, area, self.cap, s
                #pass
                
            else:
                hand_boxes = hand_boxes.cpu().numpy()
                
                im1 = im0_.copy()
            
                hand_boxes = hand_boxes[0]
                img_h = hand_boxes[3]-hand_boxes[1]
                img_w = hand_boxes[2]-hand_boxes[0]
                img_w = int(img_w)
                img_h = int(img_h)
            

                if img_w >= 1.3*img_h:
                    crop_image = im1[int(hand_boxes[1]):int(hand_boxes[3]),int(hand_boxes[0]):int(hand_boxes[2])]
                    #crop_image = im1[int(hand_boxes[1]/1.2):int(hand_boxes[3]*1.12),int(hand_boxes[0]/1.3):int(hand_boxes[2]*1.2)]
                    #crop_image = im1[int(hand_boxes[1]/1.2):int(hand_boxes[3]*1.12),int(hand_boxes[0]/1.5):int(hand_boxes[2]*1.12)]
                    #cv2.imwrite("./crop_imgs/hand_data_crop/" + img[:-4] + f"_{i}.jpg", crop_image)
                    # cv2.rectangle(im1, (int(hand_boxes[0]/1.5), int(hand_boxes[1]/1.2)), (int(hand_boxes[2]*1.12), int(hand_boxes[3]*1.12)), (0, 255, 0), 2)
                    # with open('./hand_data_boxes.txt', 'a') as f1:
                    #     f1.write(imgpath[:-4]+f"_{i}"+' '+str(np.round(np.copy(hand_boxes[0])/1.5,2))+' '+str(np.round(np.copy(hand_boxes[1])/1.2,2))+' '+str(np.round(np.copy(hand_boxes[2])*1.12,2))+' '+str(np.round(np.copy(hand_boxes[3])*1.12,2))+'\n')#call_phone
                    area = [int(hand_boxes[0]), int(hand_boxes[1]), int(hand_boxes[2]), int(hand_boxes[3])]
                    #area = [int(hand_boxes[0]/1.3), int(hand_boxes[1]/1.2), int(hand_boxes[2]/1.2), int(hand_boxes[3]/1.12)]
                    #area = [int(hand_boxes[0]/1.5), int(hand_boxes[1]/1.2), int(hand_boxes[2]*1.12), int(hand_boxes[3]*1.12)]
                    # cv2.imshow("im1",im1)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                else:
                    crop_image = im1[int(hand_boxes[1]):int(hand_boxes[3]),int(hand_boxes[0]):int(hand_boxes[2])]
                    #crop_image = im1[int(hand_boxes[1]/1.12):int(hand_boxes[3]*1.12),int(hand_boxes[0]/1.2):int(hand_boxes[2]*1.12)]
                    #rop_image = im1[int(hand_boxes[1]/1.2):int(hand_boxes[3]*1.12),int(hand_boxes[0]/1.2):int(hand_boxes[2]*1.12)]
                    #cv2.imwrite("./crop_imgs/hand_data_crop/" + img[:-4] + f"_{i}.jpg", crop_image)
                    # cv2.rectangle(im1, (int(hand_boxes[0]/1.2), int(hand_boxes[1]/1.2)), (int(hand_boxes[2]*1.12), int(hand_boxes[3]*1.12)), (0, 255, 0), 2)
                    # with open('./hand_data_boxes.txt', 'a') as f1:
                    #     f1.write(imgpath[:-4]+f"_{i}"+' '+str(np.round(np.copy(hand_boxes[0])/1.2,2))+' '+str(np.round(np.copy(hand_boxes[1])/1.2,2))+' '+str(np.round(np.copy(hand_boxes[2])*1.12,2))+' '+str(np.round(np.copy(hand_boxes[3])*1.12,2))+'\n')#call_phone
                    area = [int(hand_boxes[0]), int(hand_boxes[1]), int(hand_boxes[2]), int(hand_boxes[3])]
                    #area = [int(hand_boxes[0]/1.2), int(hand_boxes[1]/1.12), int(hand_boxes[2]*1.12), int(hand_boxes[3]*1.12)]
                    #area = [int(hand_boxes[0]/1.2), int(hand_boxes[1]/1.2), int(hand_boxes[2]*1.12), int(hand_boxes[3]*1.12)]
                    # cv2.imshow("im1",im1)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

                # print(area)
                # print((area[0]*ratio1, area[1]*ratio1), (area[2]*ratio1, area[3]*ratio1))
                # cv2.rectangle(im1, (area[0], area[1]), (area[2], area[3]), (255, 0, 255), 2)
                # cv2.imshow("im1",im1)
                # #cv2.waitKey(0)
                # #exit(0)
                # print(im0.shape)
                # print(im1.shape)

                xmin_ratio=ratio_w*area[0]
                xmax_ratio=ratio_w*area[2]
                ymin_ratio=ratio_h*area[1]
                ymax_ratio=ratio_h*area[3]

                cv2.rectangle(im0, (int(xmin_ratio), int(ymin_ratio)), (int(xmax_ratio), int(ymax_ratio)), (255, 0, 0), 2)
                # cv2.imshow("im0",im0)
                # cv2.waitKey(0)
                
                area_large = [int(xmin_ratio), int(ymin_ratio), int(xmax_ratio), int(ymax_ratio)]#扩充后区域
                crop_image_ = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)#添加单通道
               
        #----------------------------------------------------------------------------------------------------------
                if self.transforms:
                    im = self.transforms(crop_image_)  # transforms
                else:
                    im = LetterBox(self.imgsz, self.auto, stride=self.stride)(image=crop_image_)
                    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                    im = np.ascontiguousarray(im)  # contiguous
                    #im_list.append(im)
                #return path, im, im0, self.cap, s
                return path, im, im0, area_large, self.cap, s #add hand_detection
                #return path, im, im0_, area, self.cap, s#添加人脸框

    def _new_video(self, path):
        # Create a new video capture object
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)
        if hasattr(cv2, 'CAP_PROP_ORIENTATION_META'):  # cv2<4.6.0 compatibility
            self.orientation = int(self.cap.get(cv2.CAP_PROP_ORIENTATION_META))  # rotation degrees
            # Disable auto-orientation due to known issues in https://github.com/ultralytics/yolov5/issues/8493
            # self.cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)

    def _cv2_rotate(self, im):
        # Rotate a cv2 video manually
        if self.orientation == 0:
            return cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
        elif self.orientation == 180:
            return cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self.orientation == 90:
            return cv2.rotate(im, cv2.ROTATE_180)
        return im

    def __len__(self):
        return self.nf  # number of files


class LoadPilAndNumpy:

    def __init__(self, im0, imgsz=640, stride=32, auto=True, transforms=None):
        if not isinstance(im0, list):
            im0 = [im0]
        self.im0 = [self._single_check(im) for im in im0]
        self.imgsz = imgsz
        self.stride = stride
        self.auto = auto
        self.transforms = transforms
        self.mode = 'image'
        # generate fake paths
        self.paths = [f"image{i}.jpg" for i in range(len(self.im0))]
        self.bs = len(self.im0)

    @staticmethod
    def _single_check(im):
        assert isinstance(im, (Image.Image, np.ndarray)), f"Expected PIL/np.ndarray image type, but got {type(im)}"
        if isinstance(im, Image.Image):
            im = np.asarray(im)[:, :, ::-1]
            im = np.ascontiguousarray(im)  # contiguous
        return im

    def _single_preprocess(self, im, auto):
        if self.transforms:
            im = self.transforms(im)  # transforms
        else:
            im = LetterBox(self.imgsz, auto=auto, stride=self.stride)(image=im)
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous
        return im

    def __len__(self):
        return len(self.im0)

    def __next__(self):
        if self.count == 1:  # loop only once as it's batch inference
            raise StopIteration
        auto = all(x.shape == self.im0[0].shape for x in self.im0) and self.auto
        im = [self._single_preprocess(im, auto) for im in self.im0]
        im = np.stack(im, 0) if len(im) > 1 else im[0][None]
        self.count += 1
        return self.paths, im, self.im0, None, ''

    def __iter__(self):
        self.count = 0
        return self


def autocast_list(source):
    """
    Merges a list of source of different types into a list of numpy arrays or PIL images
    """
    files = []
    for im in source:
        if isinstance(im, (str, Path)):  # filename or uri
            files.append(Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im))
        elif isinstance(im, (Image.Image, np.ndarray)):  # PIL or np Image
            files.append(im)
        else:
            raise TypeError(f"type {type(im).__name__} is not a supported Ultralytics prediction source type. \n"
                            f"See https://docs.ultralytics.com/predict for supported source types.")

    return files


LOADERS = [LoadStreams, LoadPilAndNumpy, LoadImages, LoadScreenshots]

if __name__ == "__main__":
    img = cv2.imread(str(ROOT / "assets/bus.jpg"))
    dataset = LoadPilAndNumpy(im0=img)
    for d in dataset:
        print(d[0])
