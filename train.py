import tensorflow as tf
import tensorflow_addons as tfa
from model import YOLOv4
from dataloader import loader

def compute_loss(out_box,label_box):

    # TODO: GIOU
    # TODO: BBOX loss
    # TODO: cls loss
    # TODO: conf loss
    return


def train(args,hyp):
    YOLO = YOLOv4.YOLOv4(args)
    YOLO.model.compile(optimizer=tfa.optimizers.SGDW(learning_rate = 0,weight_decay=hyp['weight_decay'],momentum=hyp['momentum']),
                       loss=compute_loss)

    train_set = loader.DataLoader(args, hyp,'train')
    val_set = loader.DataLoader(args, hyp,'val')

    steps_per_epoch = len(train_set)
    global_step = tf.Variable(1, trainable = False, dype=tf.int64)
    warmup_steps = args.warmup_epoch * steps_per_epoch



    #YOLO.model.optimizer.lr.assign()

if __name__== '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Darknet53 implementation.')
    parser.add_argument('--batch_size', type=int, help = 'size of batch', default=4)
    parser.add_argument('--img_size',              type=int,   help='input height', default=512)
    parser.add_argument('--data_root',              type=str,   help='', default='./data')
    parser.add_argument('--class_file',              type=str,   help='', default='coco.names')
    parser.add_argument('--num_classes', type=int, help='', default=80)
    parser.add_argument('--augment',              action='store_false',   help='')
    parser.add_argument('--mosaic', action='store_false', help='')
    parser.add_argument('--is_shuffle', action='store_false', help='')
    parser.add_argument('--epoch', type=int,default=30 )
    parser.add_argument('--warmup_epoch', type=int, default=2)
    parser.add_argument('--mode',
                        default='train',
                        const='train',
                        nargs='?',
                        choices=['train', 'test'],
                        help='Mode [train, test]')
    args = parser.parse_args()
    hyp = {'giou': 3.54,  # giou loss gain
           'cls': 37.4,  # cls loss gain
           'cls_pw': 1.0,  # cls BCELoss positive_weight
           'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
           'obj_pw': 1.0,  # obj BCELoss positive_weight
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

    train(args,hyp)