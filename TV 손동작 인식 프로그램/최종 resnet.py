import random
import pandas as pd
import numpy as np
import os
import cv2
import mediapipe as mp

import torch

from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.models as models

from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings(action='ignore')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

CFG = {
    'FPS':30,
    'IMG_SIZE':128,
    'EPOCHS':100,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':2,
    'SEED':41
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED']) # Seed 고정

df = pd.read_csv('./train.csv')

train, val, _, _ = train_test_split(df, df['label'], test_size=0.2, random_state=CFG['SEED'])

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# 데이터 전처리
class CustomDataset(Dataset):
    def __init__(self, video_path_list, label_list):
        self.video_path_list = video_path_list
        self.label_list = label_list # 데이터 전처리
        # self.transform = transform

    def __getitem__(self, index): # 데이터셋에서 1개의 샘플을 가져오는 함수
        frames = self.get_video(self.video_path_list[index])

        if self.label_list is not None:
            label = self.label_list[index]
            return frames, label
        else:
            return frames

    def __len__(self): # 데이터셋의 길이. 총샘플수 가져오기
        return len(self.video_path_list)

    # 영상 데이터 전처리

    def get_video(self, path):
        frames = []
        cap = cv2.VideoCapture(path)
        for _ in range(CFG['FPS']):
            ret, img = cap.read()
            # base_img = cv2.imread('base_img.png')

            # 평준화, 정규화
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 3))
            img[:, :, 0] = clahe.apply(img[:, :, 0])
            img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)

            # mediapipe
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


            # 증강
            # if self.transform:
            #     augmented = self.transform(image=img)
            #     img = augmented['image']

            # mediapipe
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(img,hand_landmarks, mp_hands.HAND_CONNECTIONS)

            img = cv2.resize(img, (CFG['IMG_SIZE'], CFG['IMG_SIZE']))
            img = img / 255.

            frames.append(img)

        return torch.FloatTensor(np.array(frames)).permute(3, 0, 1, 2)

# train_transform = A.Compose([
#     # A.SmallestMaxSize(max_size=160),
#     # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=1),
#     # A.RandomCrop(height=128, width=128),
#     # A.CLAHE(p=1),
#     A.Rotate(limit=20, p=0.5, border_mode=cv2.BORDER_REPLICATE),
#     # A.RGBShift(r_shift_limit=2, g_shift_limit=2, b_shift_limit=2, p=1),
#     # A.RandomBrightnessContrast(brightness_limit=(-0.8, 0.8), contrast_limit=0, p=1),
#     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),max_pixel_value=1.0,p=1),
#     # A.pytorch.transforms.ToTensorV2()
#     ])


train_dataset = CustomDataset(train['path'].values, train['label'].values)
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

val_dataset = CustomDataset(val['path'].values, val['label'].values)
val_loader = DataLoader(val_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, block_inplanes, n_input_channels=3, conv1_t_size=7, conv1_t_stride=1, no_max_pool=False, shortcut_type='B', widen_factor=1.0, n_classes=5):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

model_depth = 50

from datetime import datetime, timezone, timedelta

# 시간 고유값
kst = timezone(timedelta(hours=9))
train_serial = datetime.now(tz=kst).strftime("%Y%m%d_%H%M%S")

# 기록 경로
RECORDER_DIR = os.path.join('results', str(model_depth) ,train_serial)
# 현재 시간 기준 폴더 생성
os.makedirs(RECORDER_DIR, exist_ok=True)


def train(model, optimizer, train_loader, val_loader, scheduler, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    best_val_score = 0
    best_model = None
    best_epoch = 0
    for epoch in range(1, CFG['EPOCHS'] + 1):
        model.train()
        train_loss = []
        for videos, labels in tqdm(iter(train_loader)):
            videos = videos.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            output = model(videos)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        _val_loss, _val_score = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        print(
            f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val F1 : [{_val_score:.5f}]')

        if scheduler is not None:
            scheduler.step(_val_score)

        if best_val_score < _val_score:
            best_val_score = _val_score
            best_model = model
            best_epoch = epoch
            print('best model found!')
            torch.save(model.state_dict(), os.path.join(RECORDER_DIR, "best-model.pt"))

    print('best F1 : ', best_val_score, ', best epoch : ', best_epoch)
    return best_model



def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    preds, trues = [], []

    with torch.no_grad():
        for videos, labels in tqdm(iter(val_loader)):
            videos = videos.to(device)
            labels = labels.to(device)

            logit = model(videos)

            loss = criterion(logit, labels)

            val_loss.append(loss.item())

            preds += logit.argmax(1).detach().cpu().numpy().tolist()
            trues += labels.detach().cpu().numpy().tolist()

        _val_loss = np.mean(val_loss)

    _val_score = f1_score(trues, preds, average='macro')
    return _val_loss, _val_score

kwargs  =  {'n_input_channels' : 3,
        'conv1_t_size' : 7,
        'conv1_t_stride' : 1,
        'no_max_pool' : False,
        'shortcut_type' : 'B',
        'widen_factor' : 1.0,
        'n_classes' : 5}

if model_depth == 10:
    model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
elif model_depth == 18:
    model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
elif model_depth == 34:
    model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
elif model_depth == 50:
    model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
elif model_depth == 101:
    model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
elif model_depth == 152:
    model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
elif model_depth == 200:
    model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

# # 텐서보드에 모델구조 쓰기

model.eval()

optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["LEARNING_RATE"] ,weight_decay=.0004)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10,
                                                                 T_mult=2, eta_min=0.00001)

infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device)

test = pd.read_csv('./test.csv')

test_dataset = CustomDataset(test['path'].values, None)
test_loader = DataLoader(test_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)


def inference(model, test_loader, device):
    model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for videos in tqdm(iter(test_loader)):
            videos = videos.to(device)

            logit = model(videos)

            preds += logit.argmax(1).detach().cpu().numpy().tolist()
    return preds

preds = inference(model, test_loader, device)

submit = pd.read_csv('./sample_submission.csv')

submit['label'] = preds
submit.head()

submit.to_csv(os.path.join(RECORDER_DIR, "best-results.csv"), index=False)

T= pd.read_csv('./answer.csv')
print(f1_score(list(T.label), preds, average='macro'))
