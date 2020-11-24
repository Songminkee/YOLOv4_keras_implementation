import os
#os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow_addons as tfa
from model import YOLOv4
from dataloader import loader
import numpy as np
import time

def test(args,hyp):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    val_set = loader.DataLoader(args, hyp, 'val')
    # val_set = tf.data.Dataset.from_generator(val_set,(tf.float32,tf.float32))
    # val_set=val_set.batch(args.batch_size).repeat(1)

    YOLO = YOLOv4.YOLOv4(args, hyp)
    if args.weight_path!='':
        print('load_model from {}'.format(args.weight_path))
        YOLO.model.load_weights(args.weight_path)



    for var in YOLO.model.trainable_variables:
        print(var)

    # @tf.function
    # def test(img,labels):
    #     with tf.GradientTape() as tape:
    #         out = YOLO.model(img)
    #         print(out)
    #         loss = YOLO.loss(labels,out)
    #     print(loss)
    #     #gradients = tape.gradient(loss, YOLO.model.trainable_variables)

    for img, labels in val_set:
        out = YOLO.model(tf.expand_dims(img,0))
        print(out)
        loss = YOLO.loss(labels,out)
        print(loss)
        # test(img,labels)
        # gradients = tape.gradient(loss, YOLO.model.trainable_variables)
        # print(gradients)
        break

if __name__== '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Darknet53 implementation.')
    parser.add_argument('--batch_size', type=int, help = 'size of batch', default=2)
    parser.add_argument('--accum_steps', type=int, default=8)
    parser.add_argument('--img_size',              type=int,   help='input height', default=512)
    parser.add_argument('--data_root',              type=str,   help='', default='./data')
    parser.add_argument('--class_file',              type=str,   help='', default='coco.names')
    parser.add_argument('--num_classes', type=int, help='', default=80)
    parser.add_argument('--augment',              action='store_false',   help='')
    parser.add_argument('--mosaic', action='store_false', help='')
    parser.add_argument('--is_shuffle', action='store_false', help='')
    parser.add_argument('--epochs', type=int,default=300 )
    parser.add_argument('--warmup_epochs', type=int, default=2)
    parser.add_argument('--soft',type=float,default=0.0)
    parser.add_argument('--log_path', type=str, default='./log')
    parser.add_argument('--summary', action='store_false')
    parser.add_argument('--summary_variable', action='store_false')
    parser.add_argument('--weight_path',type=str,default='')
    parser.add_argument('--weight_save_path', type=str, default='./weight')
    parser.add_argument('--mode',
                        default='train',
                        const='train',
                        nargs='?',
                        choices=['train', 'test'],
                        help='Mode [train, test]')
    args = parser.parse_args()
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