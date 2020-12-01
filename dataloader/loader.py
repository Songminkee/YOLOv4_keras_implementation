from util import *
import cv2
import numpy as np
import random
import math

class DataLoader(tf.keras.utils.Sequence):
    def __init__(self,args,hyp,mode='val',is_padding=True):
        self.hyp = hyp
        self.is_shuffle = args.is_shuffle
        self.batch_size = args.batch_size
        self.mode = mode # train / val / eval
        self.classes = load_class_name(args.data_root,args.class_file)
        self.num_classes = args.num_classes
        self.mosaic = args.mosaic
        self.augment = args.augment
        self.img_size = args.img_size
        self.max_label=150
        if args.data_name =='coco':
            self.images_path,self.labels_path = load_coco_image_label_files(args.data_root,mode if mode !='eval' else 'val')
        else:
            self.images_path, self.labels_path = load_image_label_files(args.data_root,args.data_name,
                                                                             mode if mode != 'eval' else 'val')

        self.indices = np.arange(len(self.images_path))
        self.is_padding = is_padding


    def load_image(self,index):
        img = cv2.imread(self.images_path[index])
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if r < 1 or (self.augment and r != 1):  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return img, (h0, w0), img.shape[:2]

    def load_label(self,index):
        if self.num_classes>1:
            label = np.array([l.strip('\n').split() for l in open(self.labels_path[index]).readlines()],np.float32)
            assert np.unique(label, axis=0).shape[0],'{} have no label'.format(self.labels_path[index])
            assert label.shape[1]==5, '{} have not efficient columns'.format(self.labels_path[index])
        else:
            label = np.array([l.strip('\n').split() for l in open(self.labels_path[index]).readlines()],np.float32)
            label = np.concatenate([np.zeros([label.shape[0],1],np.float32),label],-1)
        return label

    def letterbox(self,img, new_shape=(416, 416), color=(114, 114, 114), scaleup=False):
        # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        # if auto:  # minimum rectangle
        #     dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
        # elif scaleFill:  # stretch
        #     dw, dh = 0.0, 0.0
        #     new_unpad = new_shape
        #     ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

    def __call__(self):
        return self

    def augment_hsv(self,img, hgain=0.5, sgain=0.5, vgain=0.5):
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        dtype = img.dtype  # uint8

        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
        return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    def label_padding(self,label,ratio,pad,h,w):
        x = label.copy()
        label[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
        label[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
        label[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
        label[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]
        return label

    def xyxy2xywh(self,label):
        x = label.copy()
        label[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        label[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        label[:, 2] = x[:, 2] - x[:, 0]  # width
        label[:, 3] = x[:, 3] - x[:, 1]  # height
        return label

    def random_affine(self,img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, border=0):

        if targets is None:  # targets = [cls, xyxy]
            targets = []
        height = img.shape[0] + border * 2
        width = img.shape[1] + border * 2

        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-degrees, degrees)
        s = random.uniform(1 - scale, 1 + scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

        # Translation
        T = np.eye(3)
        T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border  # x translation (pixels)
        T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border  # y translation (pixels)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

        # Combined rotation matrix
        M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
        if (border != 0) or (M != np.eye(3)).any():  # image changed
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
        # Transform label coordinates
        n = len(targets)
        if n:
            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = (xy @ M.T)[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # reject warped points outside of image
            xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
            xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
            w = xy[:, 2] - xy[:, 0]
            h = xy[:, 3] - xy[:, 1]
            area = w * h
            area0 = (targets[:, 3] - targets[:, 1]) * (targets[:, 4] - targets[:, 2])
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
            i = (w > 4) & (h > 4) & (area / (area0 * s + 1e-16) > 0.2) & (ar < 10)

            targets = targets[i]
            targets[:, 1:5] = xy[i]

        return img, targets

    def load_mosaic(self, index):
        labels4 = []
        s = self.img_size
        xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]  # mosaic center x, y

        indices = random.sample(range(len(self.indices)),3)  # 3 additional image indices
        while index in indices:
            indices = random.sample(range(len(self.indices)),3)
        indices = [index] + indices
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.load_image(index)

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            x = self.load_label(index)
            labels = x.copy()
            if x.size > 0:  # Normalized xywh to pixel xyxy format
                labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
                labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
                labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
                labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
            labels4.append(labels)

        # Concat/clip labels
        if len(labels4):
            labels4 = np.concatenate(labels4, 0)
            # np.clip(labels4[:, 1:] - s / 2, 0, s, out=labels4[:, 1:])  # use with center crop
            np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])  # use with random_affine

        # Augment
        # img4 = img4[s // 2: int(s * 1.5), s // 2:int(s * 1.5)]  # center crop (WARNING, requires box pruning)
        img4, labels4 = self.random_affine(img4, labels4,
                                      degrees=self.hyp['degrees'],
                                      translate=self.hyp['translate'],
                                      scale=self.hyp['scale'],
                                      shear=self.hyp['shear'],
                                      border=-s // 2)  # border to remove
        return img4, labels4
    def get_anchors(self):
        self.stride = default_stride()
        self.anchor = default_anchor()
        self.anchors = make_anchor(self.stride,self.anchor)

    def on_epoch_end(self):
        self.indices = np.arange(len(self.images_path))
        if self.is_shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        now_index=index
        index = self.indices[index]
        if self.mosaic and self.mode=='train':
            img, label = self.load_mosaic(index)
        else:
            label = self.load_label(index)
            if self.is_padding:
                img,(h0,w0),(h,w) = self.load_image(index)
                img, ratio, pad = self.letterbox(img, self.img_size, scaleup=self.augment and self.mode =='train')
                if self.mode=='eval':
                    id = int(self.images_path[index].split('/')[-1].split('.')[0])
                    return np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0, 0), id, pad, (
                    h0, w0), ratio, label
                label = self.label_padding(label, ratio, pad, h, w)
            else:
                if self.mode == 'eval':
                    img = cv2.imread(self.images_path[index])
                    h0,w0,_ =img.shape
                    img = cv2.resize(np.copy(img), (self.img_size, self.img_size))
                    id = int(self.images_path[index].split('/')[-1].split('.')[0])
                    return np.expand_dims(cv2.cvtColor(img,cv2.COLOR_BGR2RGB) / 255.0,0),id,None,(h0, w0),None,label


        label[:, 1:5] = self.xyxy2xywh(label[:, 1:5])
        label[:, [2, 4]] /= img.shape[0]  # height
        label[:, [1, 3]] /= img.shape[1]  # width


        if self.augment:
            img = self.augment_hsv(img,self.hyp['hsv_h'],self.hyp['hsv_s'],self.hyp['hsv_v'])
            if random.random() < 0.5:
                img = np.fliplr(img)
                label[:,1] = 1-label[:,1]

            if random.random() < 0.5:
                img = np.flipud(img)
                label[:,2] = 1-label[:,2]

        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        label_out = np.zeros([self.max_label,5])
        label_out[:label.shape[0]]=label
        if now_index==len(self)-1:
            self.on_epoch_end()
        return img/255.0,label_out

hyp = {'giou': 3.54,  # giou loss gain
       'cls': 37.4,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.20,  # iou training threshold
       'lr0': 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)
       'lrf': 0.0005,  # final learning rate (with cos scheduler)
       'momentum': 0.937,  # SGD momentum
       'weight_decay': 0.000484,  # optimizer weight decay
       'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
       'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
       'degrees': 1.98 * 0,  # image rotation (+/- deg)
       'translate': 0.05 * 0,  # image translation (+/- fraction)
       'scale': 0.5,  # image scale (+/- gain)
       'shear': 0.641 * 0}  # image shear (+/- deg)

if __name__== '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Darknet53 implementation.')
    parser.add_argument('--batch_size', type=int, help = 'size of batch', default=4)
    parser.add_argument('--img_size',              type=int,   help='input height', default=512)
    parser.add_argument('--data_root',              type=str,   help='', default='../data')
    parser.add_argument('--class_file',              type=str,   help='', default='./coco.names')
    parser.add_argument('--num_classes', type=int, help='', default=80)
    parser.add_argument('--augment',              action='store_true',   help='')
    parser.add_argument('--mosaic', action='store_false', help='')
    parser.add_argument('--is_shuffle', action='store_false', help='')
    parser.add_argument('--mode',
                        default='val',
                        const='train',
                        nargs='?',
                        choices=['train', 'test', 'val'],
                        help='Mode [train, test, val]')
    args = parser.parse_args()

    dataset = DataLoader(args,hyp)
    import matplotlib.pyplot as plt
    print(len(dataset))
    for img,label in dataset:
        shape = img.shape
        for l in label:
            x1 = int(shape[0] * (l[1] - l[3] / 2) + 0.5)
            y1 = int(shape[1] * (l[2] - l[4] / 2) + 0.5)
            x2 = int(shape[0] * (l[1] + l[3] / 2) + 0.5)
            y2 = int(shape[1] * (l[2] + l[4] / 2) + 0.5)
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0))
        plt.imshow(img)
        plt.show()
        break
