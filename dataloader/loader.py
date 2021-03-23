import sys

sys.path.append('/mnt/1E5E8DB55E8D866D/c_yolo')
print(sys.path)
from util import *
import cv2
import numpy as np
import random
import math

class DataLoader(tf.keras.utils.Sequence):
    def __init__(self,args,hyp,mode='val'):
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
        self.is_tiny = args.is_tiny

        if self.mode=='train': 
            self.border = - self.img_size // self.hyp['border'] #
 
        if args.data_name =='coco':
            self.images_path,self.labels_path = load_coco_image_label_files(args.data_root,mode if mode !='eval' else 'val')
        else:
            self.images_path, self.labels_path = load_image_label_files(args.data_root,args.data_name,
                                                                             mode if mode != 'eval' else 'val')

        self.indices = np.arange(len(self.images_path))
        self.letter_box = args.letter_box
        if self.is_shuffle:
            self.on_epoch_end()

    def load_image(self, index):
        raw_image = tf.io.read_file(self.images_path[index])
        try:
            img = tf.io.decode_image(raw_image, 3)
        except:
            img = tf.io.decode_image(raw_image, 1)
        if img.shape[-1] == 1:
            img = tf.image.grayscale_to_rgb(img)
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if r < 1 or (self.augment and r != 1):  # always resize down, only resize up if training with augmentation
            interp = tf.image.ResizeMethod.AREA if r < 1 and not self.augment else tf.image.ResizeMethod.BILINEAR
            img = tf.cast(tf.image.resize(img, [int(h0 * r), int(w0 * r)], interp),tf.uint8)

        return img.numpy(), (h0, w0), img.shape[:2]

    def load_label(self,index):
        if self.num_classes>1:
            label = np.array([l.strip('\n').split() for l in open(self.labels_path[index]).readlines()],np.float32)
            assert np.unique(label, axis=0).shape[0],'{} have no label'.format(self.labels_path[index])
            assert label.shape[1]==5, '{} have not efficient columns'.format(self.labels_path[index])
        else:
            label = np.array([l.strip('\n').split() for l in open(self.labels_path[index]).readlines()],np.float32)
            label = np.concatenate([np.zeros([label.shape[0],1],np.float32),label],-1)
        return label

    def make_letterbox(self,img, new_shape=(416, 416), color=(114, 114, 114), scaleup=False):
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

    def random_affine(self,img, targets, degrees=10, translate=.1, scale=.1, shear=10, border=0,perspective=0.0):
        #border=0
        if targets is None:  # targets = [cls, xyxy]
            targets = []
        height = img.shape[0] + border * 2
        width = img.shape[1] + border * 2

        # Center
        C = np.eye(3)
        C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3)
        P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-degrees, degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(1 - scale, 1 + scale)
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3)
        T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)


        # Combined rotation matrix
        #M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
        M = T @ S @ R @ P @ C 
        if (border != 0) or (M != np.eye(3)).any():  # image changed
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
 
        # Transform label coordinates
        n = len(targets)
        if n:
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

             # filter candidates
            i = self.box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.10)
            targets = targets[i]
            targets[:, 1:5] = new[i]

        return img, targets

    def box_candidates(self,box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
        # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
        return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates


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

        return img4, labels4
    def get_anchors(self):
        self.stride = default_stride(is_tiny=self.is_tiny)
        self.anchor = default_anchor(is_tiny=self.is_tiny)
        self.anchors = make_anchor(self.stride,self.anchor,is_tiny=self.is_tiny)

    def on_epoch_end(self):
        self.indices = np.arange(len(self.images_path))
        if self.is_shuffle:
            np.random.shuffle(self.indices)

    def random_crop(self, image, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 1:3], axis=0), np.max(bboxes[:, 3:], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            image = image[crop_ymin : crop_ymax, crop_xmin : crop_xmax]
            
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_xmin
            bboxes[:, [2, 4]] = bboxes[:, [2, 4]] - crop_ymin

        return image, bboxes

    def random_translate(self, image, bboxes):
        if random.random() < 0.5:
            h=w=self.img_size
            max_bbox = np.concatenate([np.min(bboxes[:, 1:3], axis=0), np.max(bboxes[:, 3:], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + tx
            bboxes[:, [2, 4]] = bboxes[:, [2, 4]] + ty

        return image, bboxes

    def make_true_boxes(self,labels):
        self.get_anchors()
        num_anchor = len(self.anchors) # 3 = normal, 2 = tiny model
        sizes = np.array([self.img_size//self.stride[i] for i in range(num_anchor)]) # if img_size=416 , [52,26,13]
        trues = [np.zeros((sizes[i],sizes[i],3,5+self.num_classes)) for i in range(num_anchor)]
        # print("label",trues)
        bboxes_xywh = [np.zeros((self.max_label,4)) for _ in range(num_anchor)]
        bbox_count = np.zeros((num_anchor,))
        
        for box in labels:
            box_xywh = box[1:]
            box_cls = int(box[0])

            alpha = self.hyp['soft'] / self.num_classes
            onehot = np.full(self.num_classes,alpha,dtype=np.float32)
            onehot[box_cls] = 1.0 - alpha
            
            box_xywh_scaled = box_xywh[np.newaxis,:] * sizes[:,np.newaxis]
            
            ious = []
            exist_positive = False
            for i in range(num_anchor):
                anchors_xywh = np.zeros((3, 4))
                anchors_xywh[:, 0:2] = np.floor(box_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = self.anchors[i]
                
                iou = get_iou_loss(box_xywh_scaled[i][np.newaxis,:],anchors_xywh,method='IoU') #box_iou(box_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                ious.append(iou)
                iou_mask = iou> self.hyp['iou_t']
                
                if np.any(iou_mask):
                    xind, yind = np.floor(box_xywh_scaled[i, 0:2]).astype(np.int32)
                    
                    trues[i][yind, xind, iou_mask, :] = 0
                    trues[i][yind, xind, iou_mask, 0:4] = box_xywh * self.img_size
                    trues[i][yind, xind, iou_mask, 4:5] = 1.0
                    trues[i][yind, xind, iou_mask, 5:] = onehot

                    ind = int(bbox_count[i] % self.max_label)
                    bboxes_xywh[i][ind, :4] = box_xywh * self.img_size
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                    best_anchor_ind = np.argmax(np.array(ious).reshape(-1), axis=-1)
                    best_detect = int(best_anchor_ind / 3)
                    best_anchor = int(best_anchor_ind % 3)
                    xind, yind = np.floor(box_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                    trues[best_detect][yind, xind, best_anchor, :] = 0
                    trues[best_detect][yind, xind, best_anchor, 0:4] = box_xywh * self.img_size
                    trues[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                    trues[best_detect][yind, xind, best_anchor, 5:] = onehot

                    bbox_ind = int(bbox_count[best_detect] % self.max_label)
                    bboxes_xywh[best_detect][bbox_ind, :4] = box_xywh * self.img_size
                    bbox_count[best_detect] += 1
        if self.is_tiny:
            label_sbbox,label_lbbox = trues
            sbboxes, lbboxes = bboxes_xywh
            return label_sbbox,label_lbbox,sbboxes, lbboxes 
        else:
            label_sbbox, label_mbbox, label_lbbox = trues
            sbboxes, mbboxes, lbboxes = bboxes_xywh
            return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes
  
    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        now_index=index
        index = self.indices[index]
        mosaic = False
        if self.mode == 'train' and self.mosaic:
            img,label = self.load_mosaic(index)
            mosaic = True
        else:
            label = self.load_label(index)
            img,(h0,w0),(h,w) = self.load_image(index)
            if self.letter_box:
                img, ratio, pad = self.make_letterbox(img, self.img_size, scaleup=self.augment and self.mode =='train')
                label = self.label_padding(label, ratio, pad, h, w)
                if self.mode=='eval':
                    id = self.images_path[index].split('/')[-1].split('.')[0]
                    label[:, 1:5] = self.xyxy2xywh(label[:, 1:5])
                    label[:, [2, 4]] /= img.shape[0]  # height
                    label[:, [1, 3]] /= img.shape[1]  # width
                    
                    return np.expand_dims(img/255.0, 0), id, pad, (
                    h0, w0), ratio, label
            else:
                label = self.label_padding(label,(1,1),(0,0),h,w)
                if self.mode == 'eval':
                    label[:, 1:5] = self.xyxy2xywh(label[:, 1:5])
                    label[:, [2, 4]] /= img.shape[0]  # height
                    label[:, [1, 3]] /= img.shape[1]  # width
                    img = tf.cast(tf.image.resize(img, [int(self.img_size), int(self.img_size)], tf.image.ResizeMethod.AREA),tf.uint8)
                    id = self.images_path[index].split('/')[-1].split('.')[0]
                    return np.expand_dims(tf.cast(img,tf.float32)/255.0, 0),id,None,(h0, w0),None,label
                
        if self.mode =='train' and self.augment:
            # Augment
            if mosaic:
                img, label = self.random_affine(img, label,
                                                degrees=self.hyp['degrees'],
                                                translate=self.hyp['translate'],
                                                scale=self.hyp['scale'],
                                                shear=self.hyp['shear'],
                                                border=self.border )#if self.mosaic else 0)  # border to remove
            else:
                img, label = self.random_crop(np.copy(img), np.copy(label))
                img, label = self.random_translate(np.copy(img), np.copy(label))
        if np.sum(label[:, 1:]) == 0.:
            return self.__getitem__(index)
        label[:, 1:5] = self.xyxy2xywh(label[:, 1:5])
        label[:, [2, 4]] /= img.shape[0]  # height
        label[:, [1, 3]] /= img.shape[1]  # width

        if self.mode == 'train' and self.augment:
            
            img = self.augment_hsv(img,self.hyp['hsv_h'],self.hyp['hsv_s'],self.hyp['hsv_v'])
            if random.random() < self.hyp['flip_lr']:
                img = np.fliplr(img)
                label[:,1] = 1-label[:,1]

            if random.random() < self.hyp['flip_ud']:
                img = np.flipud(img)
                label[:,2] = 1-label[:,2]

        if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size))
        
        if self.is_tiny:
            s_label, l_label, s_box, l_box=self.make_true_boxes(label)
            if now_index==len(self)-1:
                self.on_epoch_end()
            return tf.cast(img/255.0,tf.float32), s_label, l_label, s_box, l_box
        else:
            s_label, m_label, l_label, s_box, m_box, l_box=self.make_true_boxes(label)
            if now_index==len(self)-1:        
                self.on_epoch_end()
            return tf.cast(img/255.0,tf.float32),s_label, m_label, l_label, s_box, m_box, l_box#,ol


if __name__== '__main__':
    hyp = {'giou': 1.0,#3.54,  # giou loss gain
           'cls': 1.0,#37.4,  # cls loss gain
           'cls_pw' : 1.0,
           'obj': 1.0,#83.59,  # obj loss gain (=64.3*img_size/320 if img_size != 320)
           'obj_pw' : 1.0,
           'iou_t': 0.213,  # iou training threshold
           'lr0': 0.0013,  # initial learning rate (SGD=5E-3, Adam=5E-4)
           'lrf': 0.00013,  # final learning rate (with cos scheduler)
           'momentum': 0.949,  # SGD momentum
           'fl_gamma': 2.0,  # focal loss gamma (efficientDet default is gamma=1.5)
           'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
           'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
           'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
           'degrees': 0,#1.98,#10.0,#1.98 * 0,  # image rotation (+/- deg)
           'translate':0.1, #0.1,#0.05 * 0,  # image translation (+/- fraction)
           'scale':0.5,  # image scale (+/- gain)
           'shear':0,# 0.1,#0.641 * 0}  # image shear (+/- deg)
           'ignore_threshold': 0.7,
           'border' : 2,
           'flip_lr' : 0.5,
           'flip_ud' : 0.0,
           'soft' : 0.0
           }
    import argparse
    parser = argparse.ArgumentParser(description='DataLoader Test')
    parser.add_argument('--batch_size', type=int, help = 'size of batch', default=4)
    parser.add_argument('--img_size',              type=int,   help='input height', default=416)
    parser.add_argument('--data_root',              type=str,   help='', default='/mnt/1E5E8DB55E8D866D/c_yolo/data')
    parser.add_argument('--class_file',              type=str,   help='', default='box.names')
    parser.add_argument('--num_classes', type=int, help='', default=1)
    parser.add_argument('--augment',              action='store_false',   help='')
    parser.add_argument('--mosaic', action='store_true', help='')
    parser.add_argument('--is_shuffle', action='store_true', help='')
    parser.add_argument('--data_name', default='box')

    parser.add_argument('--is_tiny',  action='store_false')
    parser.add_argument('--letter_box',action ='store_false')
    parser.add_argument('--mode',
                        default='train',
                        const='train',
                        nargs='?',
                        choices=['train', 'test', 'val'],
                        help='Mode [train, test, val]')
    args = parser.parse_args()
    
    dataset = DataLoader(args,hyp,'train')
    if args.is_tiny:
        train_set = tf.data.Dataset.from_generator(dataset,(tf.float32,tf.float32,tf.float32,tf.float32,tf.float32))
    else:
        train_set = tf.data.Dataset.from_generator(dataset,(tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32))
    train_set = train_set.batch(args.batch_size).repeat(1)
    import matplotlib.pyplot as plt
   
    if args.is_tiny:
        for img,label1,label2,box1,box2 in train_set: 
            shape = img.shape
            img=np.uint8(img[0].numpy()*255)

            print('box1',np.sum(box1[0]!=0))
            print('box2',np.sum(box2[0]!=0))
            for label in [box1[0],box2[0]]:
                for l in label:
                    if np.sum(l)==0:
                        break
                    l = np.int32(l )#* shape[1])
                    print("l", l)
                    # x1,y1,x2,y2=l
                    x1 = int( (l[0] - l[2] / 2) + 0.5) 
                    y1 = int( (l[1] - l[3] / 2) + 0.5)
                    x2 = int( (l[0] + l[2] / 2) + 0.5)
                    y2 = int( (l[1] + l[3] / 2) + 0.5)
                    img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0))
                
            plt.imshow(img)
            plt.show()
    else:
        for img,label1,label2,label3,box1,box2,box3 in train_set: 
            shape = img.shape
            img=np.uint8(img[0].numpy()*255)

            print('box1',np.sum(box1[0]!=0))
            print('box2',np.sum(box2[0]!=0))
            print('box3',np.sum(box3[0]!=0))
            for label in [box1[0],box2[0],box3[0]]:
                for l in label:
                    if np.sum(l)==0:
                        break
                    l = np.int32(l )#* shape[1])
                    print("l", l)
                    # x1,y1,x2,y2=l
                    x1 = int( (l[0] - l[2] / 2) + 0.5) 
                    y1 = int( (l[1] - l[3] / 2) + 0.5)
                    x2 = int( (l[0] + l[2] / 2) + 0.5)
                    y2 = int( (l[1] + l[3] / 2) + 0.5)
                    img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0))
                
            plt.imshow(img)
            plt.show()
