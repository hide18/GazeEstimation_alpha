import os, argparse, time, datetime
from random import shuffle

import numpy as np

import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.utils.model_zoo as model_zoo
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torchsummary import summary

import datasets_plus
from model import Gaze3inputs
from utils import gazeto3d, select_device, angular


def parse_args():
  parser = argparse.ArgumentParser(description='Gaze estimation using the Gazenet based CNN network.')
  parser.add_argument(
    '--gpu', dest='gpu_id', help='GPU device id to use [0]', default='0', type=str
  )
  parser.add_argument(
    '--arch', dest='arch', help='GC use the backbone network.', default='ResNet50', type=str
  )
  parser.add_argument(
    '--num_epochs', dest='num_epochs', help='Maximun number of training epochs.', default=50, type=int
  )
  parser.add_argument(
    '--batch_size', dest='batch_size', help='Batch size.', default=16, type=int
  )
  parser.add_argument(
    '--lr', dest='lr', help='Base learning rate.', default=0.00001, type=float
  )
  parser.add_argument(
    '--alpha', dest='alpha', help='Regression loss coefficient.', default=1, type=float
  )
  parser.add_argument(
    '--dataset', dest='dataset', help='Use dataset', default="gaze360", type=str
  )
  parser.add_argument(
    '--image_dir', dest='image_dir', help='Directory path for gaze360 images.', default='datasets/Gaze360/Image', type=str
  )
  parser.add_argument(
    '--label_dir', dest='label_dir', help='Directory path for gaze360 labels.', default='datasets/Gaze360/Label', type=str
  )
  parser.add_argument(
    '--snapshot', dest='snapshot', help='Path of pretrained models.', default='', type=str
  )
  parser.add_argument(
    '--output', dest='output', help='Path of output models.', default='output/snapshots/', type=str
  )
  parser.add_argument(
    '--valpath', dest='valpath', help='Path of validation results.', default='validation/gaze360/', type=str
  )

  args = parser.parse_args()
  return args

#Specify layers to be learned and layers not to be learned
def get_ignored_params(model):
  b = [model.conv1, model.bn1]
  for i in range(len(b)):
    for module_name, module in b[i].named_modules():
      if 'bn' in module_name:
        module.eval()
      for name, param in module.named_parameters():
        yield param

def get_non_ignored_params(model):
  b = [model.layer1, model.layer2, model.layer3, model.layer4]
  for i in range(len(b)):
    for module_name, module in b[i].named_modules():
      if 'bn' in module_name:
        module.eval()
      for name, param in module.named_parameters():
        yield param

def get_fc_params(model):
  b = [model.pitch_fc, model.yaw_fc]
  for i in range(len(b)):
    for module_name, module in b[i].named_modules():
      if 'bn' in module_name:
        module.eval()
      for name, param in module.named_parameters():
        yield param


def load_filtered_state_dict(model, snapshot):
  model_dict = model.state_dict()
  snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
  model_dict.update(snapshot)
  model.load_state_dict(model_dict)


def getArch_weights(arch, bins):
  if arch == 'ResNet18':
    model = Gaze3inputs(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], 3, bins)
    pre_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
  elif arch == 'ResNet34':
    model = Gaze3inputs(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3], 3, bins)
    pre_url = 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
  elif arch == 'ResNet101':
    model = Gaze3inputs(torchvision.models.resnet.Botteleneck, [3, 4, 23, 3], 3, bins)
    pre_url = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
  elif arch == 'ResNet152':
    model = Gaze3inputs(torchvision.models.resnet.Botteleneck, [3, 8, 36, 3], 3, bins)
    pre_url = 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
  else:
    model = Gaze3inputs(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 3, bins)
    pre_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'

  return model, pre_url




if __name__=='__main__':
  args = parse_args()

  cudnn.enabled = True
  num_epochs = args.num_epochs
  batch_size = args.batch_size
  gpu = select_device(args.gpu_id, batch_size=args.batch_size)
  dataset = args.dataset
  alpha = args.alpha
  valpath = args.valpath
  output = args.output

  transformation_face = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])

  transformation_eye = transforms.Compose([
    transforms.Resize((36, 60)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])


  if dataset=="gaze360":
    model, pre_url = getArch_weights(args.arch, 180)
    if args.snapshot == '':
      face = model.face_res
      eye = model.eye_res
      load_filtered_state_dict(face, model_zoo.load_url(pre_url))
      load_filtered_state_dict(eye, model_zoo.load_url(pre_url))
    else:
        saved_state_dict = torch.load(args.snapshot)
        model.load_state_dict(saved_state_dict)
    model.cuda(gpu)
    print('Loading data.')

    label_path = args.label_dir

    #traindata dataloader
    train_label = os.path.join(label_path, "train.label")
    train_dataset = datasets_plus.Gaze360(train_label, args.image_dir, transformation_face, transformation_eye, 180, 2)
    train_loader = DataLoader(
      dataset=train_dataset,
      batch_size=int(batch_size),
      shuffle=True,
      num_workers=8,
      pin_memory=True
    )

    #validation dataloader
    val_label = os.path.join(label_path, "val.label")
    val_dataset = datasets_plus.Gaze360(val_label, args.image_dir, transformation_face, transformation_eye, 180, 2, train=False)
    val_loader = DataLoader(
      dataset=val_dataset,
      batch_size=int(batch_size),
      shuffle=False,
      num_workers=8,
      pin_memory=True
    )

    torch.backends.cudnn.benchmark = True

    today = datetime.datetime.fromtimestamp(time.time())
    summary_name = '{}_{}'.format('GN-gaze360', str(today.strftime('%Y-%-m*%-d_%-H*%-M*%-S')))

    output = os.path.join(output, summary_name)
    if not os.path.exists(output):
      os.makedirs(output)

    valpath = os.path.join(valpath, summary_name)
    if not os.path.exists(valpath):
      os.makedirs(valpath)


    criterion = nn.CrossEntropyLoss().cuda(gpu)
    reg_criterion = nn.MSELoss().cuda(gpu)
    softmax = nn.Softmax(dim=1).cuda(gpu)

    optimizer_gaze = torch.optim.Adam([
      {'params' : get_ignored_params(face), 'lr' : 0},
      {'params' : get_ignored_params(eye), 'lr' : 0},
      {'params' : get_non_ignored_params(face), 'lr' : args.lr},
      {'params' : get_non_ignored_params(eye), 'lr' : args.lr},
      {'params' : get_fc_params(model), 'lr' : args.lr}
    ], lr = args.lr)


    idx_tensor = [idx for idx in range(180)]
    idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(gpu)


    print('Ready to train and validation network.')
    configuration = f"\ntrain_validation configuration, gpu_id={args.gpu_id}, batch_size={batch_size}, model_arch={args.arch}\n"

    epoch_list = []
    avg_MAE = []

    with open(os.path.join(valpath, dataset+".log"), 'w') as outfile:
      outfile.write(configuration)
      for epoch in range(num_epochs):
        sum_loss_pitch = sum_loss_yaw = iter_gaze = 0

        #train
        model.train()
        for i, (face, left, right, labels, cont_labels, name) in enumerate(train_loader):
          #input image
          face = Variable(face).cuda(gpu)
          left = Variable(left).cuda(gpu)
          right = Variable(right).cuda(gpu)

          label_pitch = Variable(labels[:, 0]).cuda(gpu)
          label_yaw = Variable(labels[:, 1]).cuda(gpu)
          label_pitch_cont = Variable(cont_labels[:, 0]).cuda(gpu)
          label_yaw_cont = Variable(cont_labels[:, 1]).cuda(gpu)

          pitch, yaw = model(face, left, right)

          #Cross Entropy Loss
          loss_pitch = criterion(pitch, label_pitch)
          loss_yaw = criterion(yaw, label_yaw)

          pre_pitch = softmax(pitch)
          pre_yaw = softmax(yaw)
          pre_pitch = torch.sum(pre_pitch * idx_tensor, 1) * 2 - 180
          pre_yaw = torch.sum(pre_yaw * idx_tensor, 1) * 2 - 180

          #MSE Loss
          loss_cont_pitch = reg_criterion(pre_pitch, label_pitch_cont)
          loss_cont_yaw = reg_criterion(pre_yaw, label_yaw_cont)

          #Total Loss
          loss_pitch += alpha * loss_cont_pitch
          loss_yaw += alpha * loss_cont_yaw

          sum_loss_pitch += loss_pitch
          sum_loss_yaw += loss_yaw

          loss_seq = [loss_pitch, loss_yaw]
          grad_seq = [torch.tensor(1.0).cuda(gpu) for _ in range(len(loss_seq))]
          optimizer_gaze.zero_grad(set_to_none=True)
          torch.autograd.backward(loss_seq, grad_seq)
          optimizer_gaze.step()

          iter_gaze += 1

          if (i+1) % 100 == 0:
            print('Epoch [%d/%d], Iter [%d/%d], Losses : Gaze Pitch %.4f, Gaze Yaw %.4f' %
            (epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, sum_loss_pitch/iter_gaze, sum_loss_yaw/iter_gaze)
            )

        #validation
        total = 0
        avg_error = 0.0
        model.eval()
        with torch.no_grad():
          for j, (face, left, right, labels, cont_labels, name) in enumerate(val_loader):
            face = Variable(face).cuda(gpu)
            left = Variable(left).cuda(gpu)
            right = Variable(right).cuda(gpu)
            total += cont_labels.size(0)

            label_pitch = cont_labels[:, 0].float() * np.pi / 180
            label_yaw = cont_labels[:, 1].float() * np.pi / 180

            pitch, yaw = model(face, left, right)

            pre_pitch = softmax(pitch)
            pre_yaw = softmax(yaw)
            pre_pitch = torch.sum(pre_pitch * idx_tensor, 1).cpu() * 2 - 180
            pre_yaw = torch.sum(pre_yaw * idx_tensor, 1).cpu() * 2 - 180

            pitch_predicted = pre_pitch * np.pi / 180
            yaw_predicted = pre_yaw * np.pi / 180

            for p, y, pl, yl in zip(pitch_predicted, yaw_predicted, label_pitch, label_yaw):
              avg_error += angular(gazeto3d([p, y]), gazeto3d([pl, yl]))

        x = epoch + 1
        epoch_list.append(x)
        avg_MAE.append(avg_error/total)
        loger = f"---VAL--- Epoch [{x}/{num_epochs}], MAE : {avg_error/total}\n"
        print(loger)
        outfile.write(loger)

        if epoch % 1 == 0 and epoch < num_epochs:
          if torch.save(model.state_dict(), output +'/'+'_epoch_'+str(epoch+1)+'.pkl') == None:
            print('Taking snapshot... success')
