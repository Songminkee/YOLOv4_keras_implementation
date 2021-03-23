from model import YOLOv4
from util import *


def freeze_all(model, frozen=True):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_all(l, frozen)

def save(args,hyp):
    if args.is_tiny:
        YOLO = YOLOv4.YOLOv4_tiny(args,hyp)
    else:
        YOLO = YOLOv4.YOLOv4(args,hyp)

    if args.weight_path!='':
        if args.is_darknet_weight:
            print('load darkent weight from {}'.format(args.weight_path))
            load_darknet_weights(YOLO.model,args.weight_path,args.is_tiny)
        else:
            print('load_model from {}'.format(args.weight_path))
            YOLO.model.load_weights(args.weight_path).expect_partial()
    xywh,cls = get_decoded_pred(YOLO)
    model = tf.keras.Model(YOLO.backbone.input, tf.concat([xywh,cls],-1))
    model.summary()
    freeze_all(model, True)
    model.save(args.out_path)

if __name__== '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='YOLOv4 Test')
    parser.add_argument('--img_size',              type=int,   help='Size of input image / default : 416', default=416)
    parser.add_argument('--data_root',              type=str,   help='Root path of class name file and coco_%2017.txt / default : "./data"', default='./data')
    parser.add_argument('--class_file',              type=str,   help='Class name file / default :"coco.names"', default='coco.names')
    parser.add_argument('--num_classes', type=int, help='Number of classes (in COCO 80) ', default=80)
    parser.add_argument('--weight_path' ,type=str ,default='dark_weight/yolov4.weights', help='path of weight')
    parser.add_argument('--is_darknet_weight', action='store_true', help = 'If true, load the weight file used by the darknet framework.')
    parser.add_argument('--is_tiny', action='store_true', help = 'Flag for using tiny. / default : false')
    parser.add_argument('--confidence_threshold', type=float, default=0.001)
    parser.add_argument('--iou_threshold', type=float, default=0.1)
    parser.add_argument('--score_threshold', type=float, default=0.1)
    parser.add_argument('--out_path', type=str, default='./saved_model/model')
    args = parser.parse_args()
    args.batch_size= 1
    args.mode = 'eval'

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
           'translate':0.5, #0.1,#0.05 * 0,  # image translation (+/- fraction)
           'scale':0.1,  # image scale (+/- gain)
           'shear':0,# 0.1,#0.641 * 0}  # image shear (+/- deg)
           'ignore_threshold': 0.7,
           'border' : 2,
           'flip_lr' : 0.5,
           'flip_ud' : 0.0,
           'soft' : 0.0
           }
    save(args,hyp)