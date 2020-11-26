import os
#os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from model import YOLOv4
from dataloader import loader
from util import *

def test(args,hyp):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    test_set = loader.DataLoader(args, hyp, 'test')

    if args.is_tiny:
        YOLO = YOLOv4.YOLOv4_tiny(args,hyp)
    else:
        YOLO = YOLOv4.YOLOv4(args, hyp)

    if args.weight_path!='':
        if args.is_darknet_weight:
            print('load darkent weight from {}'.format(args.weight_path))
            load_darknet_weights(YOLO.model,args.weight_path,args.is_tiny)
        else:
            print('load_model from {}'.format(args.weight_path))
            YOLO.model.load_weights(args.weight_path)

    import cv2
    for img in test_set:
        pimg = img[0].copy()

        boxes, scores, classes, valid_detections = inference(YOLO,img,args)

        import matplotlib.pyplot as plt
        for i in range(valid_detections[0].numpy()):
            y_min,x_min,y_max,x_max=boxes[0][i]
            h,w,_ = pimg.shape
            pimg = cv2.rectangle(pimg,(int(x_min*w),int(y_min*h)),(int(x_max*w),int(y_max*h)),(0,255,0),1)
        plt.imshow(pimg)
        plt.show()

if __name__== '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='YOLOv4 Test')
    parser.add_argument('--batch_size', type=int, help = 'size of batch', default=1)
    parser.add_argument('--img_size',              type=int,   help='input height', default=416)
    parser.add_argument('--data_root',              type=str,   help='', default='./data')
    parser.add_argument('--class_file',              type=str,   help='', default='coco.names')
    parser.add_argument('--num_classes', type=int, help='', default=80)
    parser.add_argument('--augment',              action='store_true',   help='')
    parser.add_argument('--mosaic', action='store_true', help='')
    parser.add_argument('--is_shuffle', action='store_false', help='')
    parser.add_argument('--weight_path',type=str,default='dark_weight/yolov4-tiny.weights')
    parser.add_argument('--is_darknet_weight', action='store_false')
    parser.add_argument('--is_tiny', action='store_false')
    parser.add_argument('--confidence_threshold', type=float, default=0.2)
    parser.add_argument('--iou_threshold', type=float, default=0.5)
    parser.add_argument('--score_threshold', type=float, default=0.25)
    args = parser.parse_args()

    args.mode='test'
    args.soft = 0.0

    hyp = {'giou': 3.54,  # giou loss gain
           'cls': 37.4,  # cls loss gain
           'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
           'iou_t': 0.213,  # iou training threshold
           'lr0': 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)
           'lrf': 0.0005,  # final learning rate (with cos scheduler)
           'momentum': 0.949,  # SGD momentum
           'weight_decay': 0.000484,  # optimizer weight decay
           'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
           'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
           'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
           'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
           'degrees': 1.98 * 0,  # image rotation (+/- deg)
           'translate': 0.05 * 0,  # image translation (+/- fraction)
           'scale': 0.5,  # image scale (+/- gain)
           'shear': 0.641 * 0}  # image shear (+/- deg)

    test(args,hyp)