import os
#os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from model import YOLOv4
import glob
from util import *
import cv2



def detect(image,YOLO,class_name,args):
    image = np.squeeze(image)
    img = image.copy()
    h, w, _ = img.shape
    if h != args.img_size or w != args.img_size:
        img = cv2.resize(img,(args.img_size,args.img_size))

    boxes, scores, classes, valid_detections = inference(YOLO, img[np.newaxis, :, :, :], args)
    scores, classes, valid_detections = np.float32(np.squeeze(scores)), np.int32(np.squeeze(classes)),np.int32(np.squeeze(valid_detections))
    bbox_thick = int(0.6 * (h + w) / 600)

    fontScale = 0.5
    box_color = (0,255,0)
    y_min, x_min, y_max, x_max = convert_to_origin_shape(boxes[0], None, None, h, w)
    y_min, x_min, y_max, x_max = np.float32(np.squeeze(y_min)), np.float32(np.squeeze(x_min)),np.float32(np.squeeze(y_max)),np.float32(np.squeeze(x_max))
    for i in range(valid_detections):
        if scores[i] < args.score_threshold:
            break

        # draw rectangle
        image = cv2.rectangle(image,(x_min[i],y_min[i]),(x_max[i],y_max[i]),box_color,thickness = 2)

        # draw text
        bbox_mess = '%s: %.2f' % (class_name[classes[i]], scores[i])

        t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
        t = (x_min[i] + t_size[0], y_min[i] - t_size[1] - 3)
        image = cv2.rectangle(image, (x_min[i],y_min[i]), (np.float32(t[0]), np.float32(t[1])), box_color, -1)  # filled
        image = cv2.putText(image, bbox_mess, (int(x_min[i]), int(y_min[i] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
    return image

def detect_example(args,hyp):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

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

    class_name = load_class_name(args.data_root, args.class_file)
    image_pathes = glob.glob(os.path.join(args.input_dir,'*'))

    for im_path in image_pathes:
        img = cv2.cvtColor(cv2.imread(im_path),cv2.COLOR_BGR2RGB)/255.0
        img = cv2.cvtColor(detect(img,YOLO,class_name,args),cv2.COLOR_RGB2BGR)
        cv2.imshow('detected',img)
        cv2.waitKey()



if __name__== '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='YOLOv4 Test')
    parser.add_argument('--img_size',              type=int,   help='Size of input image / default : 416', default=416)
    parser.add_argument('--data_root',              type=str,   help='Root path of class name file and coco_%2017.txt / default : "./data"', default='./data')
    parser.add_argument('--class_file',              type=str,   help='Class name file / default : "coco.name"', default='box.names') # 'coco.names'
    parser.add_argument('--num_classes', type=int, help='Number of classes (in COCO 80) ', default=1) # 80
    parser.add_argument('--weight_path',type=str,default='weight/box/final', help='path of weight') # 'dark_weight/yolov4.weights'
    parser.add_argument('--is_darknet_weight', action='store_true', help = 'If true, load the weight file used by the darknet framework.') # 'store_false'
    parser.add_argument('--is_tiny', action='store_true', help = 'Flag for using tiny. / default : false')
    parser.add_argument('--input_dir',type=str,default='./data/dataset/box_dataset/images/val')
    parser.add_argument('--confidence_threshold', type=float, default=0.001)
    parser.add_argument('--iou_threshold', type=float, default=0.6)
    parser.add_argument('--score_threshold', type=float, default=0.6)
    args = parser.parse_args()

    args.mode='eval'
    args.soft = 0.0
    args.batch_size = 1

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


    detect_example(args,hyp)