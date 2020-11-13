from util import *
import cv2
import numpy as np

class DataLoader(object):
    def __init__(self,args):
        self.batch_size = args.batch_size
        self.mode = args.mode # train / test / val
        self.classes = load_class_name(args.data_root,args.class_file)
        self.num_classes = args.num_classes
        self.mosaic = args.mosaic
        self.augment = args.augment
        self.img_size = args.img_size
        self.images_path,self.labels_path = load_coco_image_label_files(args.data_root,args.mode)

    def load_image(self,index):
        img = cv2.imread(self.images_path[index])
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if r < 1 or (self.augment and r != 1):  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return img, (h0, w0), img.shape[:2]

    def load_label(self,index):
        label = np.array([l.strip('\n').split() for l in open(self.labels_path[index]).readlines()],np.float32)
        assert np.unique(label, axis=0).shape[0],'{} have no label'.format(self.labels_path[index])
        assert label.shape[1]==5, '{} have not efficient columns'.format(self.labels_path[index])
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

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        img,(h0,w0),(h,w) = self.load_image(index)
        label = self.load_label(index)
        img, ratio, pad = self.letterbox(img, self.img_size, scaleup=self.augment)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling
        x = label.copy()
        label[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
        label[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
        label[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
        label[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]

        return img, ratio,pad,shapes


if __name__== '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Darknet53 implementation.')
    parser.add_argument('--batch_size', type=int, help = 'size of batch', default=4)
    parser.add_argument('--img_size',              type=int,   help='input height', default=512)
    parser.add_argument('--data_root',              type=str,   help='', default='../data')
    parser.add_argument('--class_file',              type=str,   help='', default='./coco.names')
    parser.add_argument('--num_classes', type=int, help='', default=80)
    parser.add_argument('--augment',              action='store_true',   help='')
    parser.add_argument('--mosaic', action='store_true', help='')
    parser.add_argument('--mode',
                        default='val',
                        const='train',
                        nargs='?',
                        choices=['train', 'test', 'val'],
                        help='Mode [train, test, val]')
    args = parser.parse_args()

    dataset = DataLoader(args)
    import matplotlib.pyplot as plt

    for img,(h0,w0),(h,w) in dataset:
        print(type(img))
        plt.imshow(img)
        plt.show()
        break

