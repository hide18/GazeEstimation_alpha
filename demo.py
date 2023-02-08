import argparse
import numpy as np
import cv2
import time
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

from PIL import Image
from utils import select_device, draw_gaze
from PIL import Image, ImageOps

from face_detection import RetinaFace
from model import Gaze3inputs


eye_cascade_path = './haarcascade_eye.xml'
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze evalution using model pretrained with L2CS-Net on Gaze360.')
    parser.add_argument(
        '--gpu',dest='gpu_id', help='GPU device id to use [0]',
        default="0", type=str)
    parser.add_argument(
        '--snapshot',dest='snapshot', help='Path of model snapshot.',
        default='output/snapshots/L2CS-gaze360-_loader-180-4/_epoch_55.pkl', type=str)
    parser.add_argument(
        '--cam',dest='cam_id', help='Camera device id to use [0]',
        default=0, type=int)
    parser.add_argument(
        '--arch',dest='arch',help='Network architecture, can be: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152',
        default='ResNet50', type=str)

    args = parser.parse_args()
    return args

def getArch(arch, bins):
  if arch == 'ResNet18':
    model = Gaze3inputs(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], 3, bins)
  elif arch == 'ResNet34':
    model = Gaze3inputs(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3], 3, bins)
  elif arch == 'ResNet101':
    model = Gaze3inputs(torchvision.models.resnet.Botteleneck, [3, 4, 23, 3], 3, bins)
  elif arch == 'ResNet152':
    model = Gaze3inputs(torchvision.models.resnet.Botteleneck, [3, 8, 36, 3], 3, bins)
  else:
    model = Gaze3inputs(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 3, bins)

  return model

def eye_place(eyes):
  return eyes[0], eyes[1], eyes[2], eyes[3]


if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    arch=args.arch
    batch_size = 1
    cam = args.cam_id
    gpu = select_device(args.gpu_id, batch_size=batch_size)
    snapshot_path = args.snapshot

    transformation_face = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])

    transformation_eye = transforms.Compose([
    transforms.Resize((60, 60)),
    transforms.CenterCrop((36, 60)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])

    model=getArch(arch, 180)
    print('Loading snapshot.')
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)
    model.cuda(gpu)
    model.eval()


    softmax = nn.Softmax(dim=1)
    detector = RetinaFace(gpu_id=0)
    idx_tensor = [idx for idx in range(180)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
    x=0

    cap = cv2.VideoCapture(cam)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    with torch.no_grad():
        while True:
            success, frame = cap.read()
            start_fps = time.time()

            faces = detector(frame)
            if faces is not None:
                for box, landmarks, score in faces:
                    if score < .95:
                        continue
                    x_min=int(box[0])
                    if x_min < 0:
                        x_min = 0
                    y_min=int(box[1])
                    if y_min < 0:
                        y_min = 0
                    x_max=int(box[2])
                    y_max=int(box[3])
                    bbox_width = x_max - x_min
                    bbox_height = y_max - y_min
                    # x_min = max(0,x_min-int(0.2*bbox_height))
                    # y_min = max(0,y_min-int(0.2*bbox_width))
                    # x_max = x_max+int(0.2*bbox_height)
                    # y_max = y_max+int(0.2*bbox_width)
                    # bbox_width = x_max - x_min
                    # bbox_height = y_max - y_min

                    # Crop image
                    img = frame[y_min:y_max, x_min:x_max]
                    img = cv2.resize(img, (224, 224))
                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    eyes = eye_cascade.detectMultiScale(img_gray, scaleFactor=1.01, minNeighbors=80, minSize=(12, 20))
                    if len(eyes) != 2:
                      cv2.putText(frame, 'not able to detect eyes.', (100, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(0, 0, 255), thickness=2)
                    else:
                      right_place = eyes[0]
                      left_place = eyes[1]
                      ex, ey, ew, eh = eye_place(right_place)
                      right = img[ey:ey+eh, ex:ex+ew]
                      right = cv2.resize(right, (60, 60))
                      ex, ey, ew, eh = eye_place(left_place)
                      left = img[ey:ey+eh, ex:ex+ew]
                      left = cv2.resize(left, (60, 60))

                      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                      right = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)
                      left = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
                      im_pil = Image.fromarray(img)
                      right_pil = Image.fromarray(right)
                      left_pil = Image.fromarray(left)
                      img=transformation_face(im_pil)
                      left=transformation_eye(left_pil)
                      right=transformation_eye(right_pil)

                      img  = Variable(img).cuda(gpu)
                      img  = img.unsqueeze(0)
                      left  = Variable(left).cuda(gpu)
                      left  = left.unsqueeze(0)
                      right  = Variable(right).cuda(gpu)
                      right  = right.unsqueeze(0)

                      # gaze prediction
                      gaze_pitch, gaze_yaw = model(img, left, right)


                      pitch_predicted = softmax(gaze_pitch)
                      yaw_predicted = softmax(gaze_yaw)

                      # Get continuous predictions in degrees.
                      pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 2 - 180
                      yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 2 - 180

                      pitch_predicted= pitch_predicted.cpu().detach().numpy()* np.pi/180.0
                      yaw_predicted= yaw_predicted.cpu().detach().numpy()* np.pi/180.0



                      draw_gaze(x_min,y_min,bbox_width, bbox_height,frame,(pitch_predicted,yaw_predicted),color=(0,0,255))
                      cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)
            myFPS = 1.0 / (time.time() - start_fps)
            cv2.putText(frame, 'FPS: {:.1f}'.format(myFPS), (10, 20),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow("Demo",frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            success,frame = cap.read()
