from model import YOLOv4
from util import *


def freeze_all(model, frozen=True):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_all(l, frozen)

def save(args):

    if args.is_tiny:
        YOLO = YOLOv4.YOLOv4_tiny(args)
    else:
        YOLO = YOLOv4.YOLOv4(args)

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
    args.mode = 'test'
    save(args)