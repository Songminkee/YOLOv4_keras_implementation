import os
#os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from model import YOLOv4
from dataloader import loader
from util import *
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def test_coco(args,hyp,test_set):
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
            YOLO.model.load_weights(args.weight_path).expect_partial()

    names = test_set.classes
    json_list = []
    yolo2coco = coco80_to_coco91_class()
    result = []
    cnt = 0

    for img,id,pad,(h0,w0),ratio,label in test_set:
        h, w, _ = img[0].shape

        xywh, cls = decode(YOLO, img)
        boxes, scores, classes, valid_detections = inference(xywh, cls, args)
        label[:,1:] = scaled_xywh2xyxy(label[:,1:],h0,w0)

        positive=np.zeros((valid_detections[0].numpy()),dtype=np.bool)
        detected = []

        y_min, x_min, y_max, x_max = convert_to_origin_shape(boxes[0], pad, ratio, h0, w0, h, w, args.letter_box)

        n_box = tf.concat([x_min, y_min, x_max, y_max], -1)
        
        ious = get_iou_for_test(n_box, label[:, 1:])
        for i in range(valid_detections[0].numpy()):
            json_list.append({'image_id': int(id),
                              'category_id': yolo2coco[int(classes[0][i])],
                              'bbox': [float(round(b.numpy(), 3)) for b in
                                       [x_min[i][0], y_min[i][0], x_max[i][0] - x_min[i][0],
                                        y_max[i][0] - y_min[i][0]]],
                              'score': float(round(scores[0][i].numpy(), 5))})

            mask = tf.cast(label[:,0],tf.int8) == tf.cast(classes[0][i],tf.int8)
            max_iou, max_idx = tf.reduce_max(tf.where(mask,ious[i],0),-1),tf.argmax(tf.where(mask,ious[i],0),-1)
            max_iou, max_idx = max_iou.numpy(),max_idx.numpy()
            
            if max_iou>=0.5 and max_idx not in detected:
                positive[i]=True
                detected.append(max_idx)

        result.append((positive,scores[0][:valid_detections[0].numpy()],classes[0][:valid_detections[0].numpy()],label[:,0]))

        cnt+=1
        print(cnt,'/',len(test_set))
        if cnt==len(test_set):
            break

    result = [np.concatenate(x, 0) for x in zip(*result)]
    p, r, ap, f1, ap_class = ap_per_class(*result)
    mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
    nt = np.bincount(result[3].astype(np.int64), minlength=args.num_classes)

    # Print results
    pf = '%20s' + '%10.3g' * 5  # print format
    pc = '%20s' +'%10.5s'+'%10.3s' * 4
    print(pc % ('','Total','mP','mR','mAP','mF1'))
    print(pf % ('all', nt.sum(), mp, mr, map, mf1),'\n')
    print(pc % ('Class', 'Total', 'P', 'R', 'AP', 'F1'))
    for i, c in enumerate(ap_class):
        print(pf % (names[c], nt[c], p[i], r[i], ap[i], f1[i]))

    # COCO eval
    if not os.path.exists(args.out_json_path):
        os.makedirs(args.out_json_path)
    with open(os.path.join(args.out_json_path,'results4.json'), 'w') as file:
        json.dump(json_list, file)

    map50,map = coco_eval(os.path.join(args.out_json_path,'results4.json'),test_set)
    return mp, mr, map50, map

def test(args,hyp,test_set):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if args.is_tiny:
        YOLO = YOLOv4.YOLOv4_tiny(args, hyp)
    else:
        YOLO = YOLOv4.YOLOv4(args, hyp)

    if args.weight_path != '':
        if args.is_darknet_weight:
            print('load darkent weight from {}'.format(args.weight_path))
            load_darknet_weights(YOLO.model, args.weight_path, args.is_tiny)
        else:
            print('load_model from {}'.format(args.weight_path))
            YOLO.model.load_weights(args.weight_path).expect_partial()

    names = test_set.classes
    result = []
    cnt = 0
    import cv2
    for img, _, pad, (h0, w0), ratio, label in test_set:
        h, w, _ = img[0].shape
        #print(img[0])
        xywh,cls = decode(YOLO,img)
        boxes, scores, classes, valid_detections = inference(xywh, cls, args)
              
        label[:, 1:] = scaled_xywh2xyxy(label[:, 1:], h0, w0)
        positive = np.zeros((valid_detections[0].numpy(),10), dtype=np.bool)
        detected = []
        y_min, x_min, y_max, x_max = convert_to_origin_shape(boxes[0], pad, ratio, h0, w0, h, w, args.letter_box)
        
        n_box = tf.concat([x_min, y_min, x_max, y_max], -1)
        ious = get_iou_for_test(n_box,label[:,1:])
        
        for i in range(valid_detections[0].numpy()):
            mask = tf.cast(label[:, 0], tf.int8) == tf.cast(classes[0][i], tf.int8)
            max_iou, max_idx = tf.reduce_max(tf.where(mask, ious[i], 0), -1), tf.argmax(tf.where(mask, ious[i], 0), -1)
            max_iou, max_idx = max_iou.numpy(), max_idx.numpy()
            if max_iou < 0.5: continue
            if max_idx not in detected:
                # print(max_iou)
                # print(np.linspace(0.5,0.95,10))
                positive[i] = max_iou >= np.linspace(0.5,0.95,10)
                detected.append(max_idx)

        result.append(
            (positive, scores[0][:valid_detections[0].numpy()], classes[0][:valid_detections[0].numpy()], label[:, 0]))

        cnt += 1
        print(cnt, '/', len(test_set))
        if cnt == len(test_set):
            break

    result = [np.concatenate(x, 0) for x in zip(*result)]
    p, r, ap, f1, ap_class = ap_per_class(*result)
    p, r,f1 = p[:,0], r[:,0],f1[:,0]
    mp, mr, map50,map, mf1 = p.mean(), r.mean(), ap[:,0].mean(), ap.mean(1), f1.mean()
    nt = np.bincount(result[3].astype(np.int64), minlength=args.num_classes)
    ap = ap[:,0]
    # Print results
    if not os.path.exists(args.weight_path.replace('.','result')):
            os.makedirs(args.weight_path.replace('.','result'))
    f = open(args.weight_path.replace('.','result')+'/ret.txt','w')
    
    pf = '%10s' + '%13.3g' * 6  # print format
    pc = '%10s' + '%13.5s' + '%13.10s' * 5
    pf2 = '%10s' + '%13.3g' * 5
    pc2 = '%10s' + '%13.5s' + '%13.5s' * 4
    print(pc % ('', 'Total', 'mP', 'mR', 'mAP@50','mAP@.5:.95', 'mF1'))
    print(pf % ('all', nt.sum(), mp, mr, map50, map, mf1), '\n')
    print(pc2 % ('Class', 'Total', 'P', 'R', 'AP@50', 'F1'))
    f.writelines(pc % ('', 'Total', 'mP', 'mR', 'mAP@50','mAP@50:90', 'mF1')+'\n')
    f.writelines(pf % ('all', nt.sum(), mp, mr, map50, map, mf1)+ '\n')
    f.writelines(pc2 % ('Class', 'Total', 'P', 'R', 'AP@50', 'F1'))
    for i, c in enumerate(ap_class):
        print(pf2 % (names[c], nt[c], p[i], r[i], ap[i], f1[i]))
        f.writelines(pf2 % (names[c], nt[c], p[i], r[i], ap[i], f1[i])+'\n')
    f.close()
    return mp, mr


def coco_eval(json_path,test_set):
    cocoGt = COCO(glob.glob(args.annotation_path+'/instances_val*.json')[0])  # initialize COCO ground truth api
    cocoDt = cocoGt.loadRes(json_path)  # initialize COCO pred api

    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.params.imgIds = [int(path.split('/')[-1].split('.')[0]) for path in test_set.images_path]  # [:32]  # only evaluate these images
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    map, map50 = cocoEval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)

    return map50, map

if __name__== '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='YOLOv4 Test')
    parser.add_argument('--img_size',              type=int,   help='Size of input image / default : 416', default=416)
    parser.add_argument('--data_root',              type=str,   help='Root path of class name file and coco_%2017.txt / default : "./data"', default='./data')
    parser.add_argument('--class_file',              type=str,   help='Class name file / default : "coco.name"', default='coco.names')
    parser.add_argument('--num_classes', type=int, help='Number of classes (in COCO 80) ', default=80)
    parser.add_argument('--data_name', type=str,
                        help='Root path of class name file and coco_%2017.txt / default : "./data"'
                        , default='coco')
    parser.add_argument('--augment',              action='store_true',   help='Flag of augmentation (hsv, flip, random affine) / default : false')
    parser.add_argument('--mosaic', action='store_true', help='Flag of mosaic augmentation / default : false')
    parser.add_argument('--is_shuffle', action='store_true', help='Flag of data shuffle / default : true')
    parser.add_argument('--only_coco_eval', action='store_true', help=' When the flag is true, only pycocotools is used to show the result. In this case, you must enter the path to the json file. / default : false')
    parser.add_argument('--out_json_path',type=str,default='./eval',help= 'Folder of output file (json) / default : "./eval"')
    parser.add_argument('--json_path',type=str,default='./eval/results2.json', help = 'Path of json result file. This flag only used when the only_coco_eval flag is true' )
    parser.add_argument('--weight_path',type=str,default='dark_weight/yolov4.weights', help='path of weight')
    parser.add_argument('--annotation_path', type=str, default='./data/dataset/COCO/annotations',help= 'COCO annotation file folder / default : "./data/dataset/COCO/annotations"')
    parser.add_argument('--is_darknet_weight', action='store_true', help = 'If true, load the weight file used by the darknet framework.')
    parser.add_argument('--is_tiny', action='store_true', help = 'Flag for using tiny. / default : false')
    parser.add_argument('--is_tflite', action='store_true', help='Flag for using tflite. / default : false')
    parser.add_argument('--is_coco', action='store_true', help='Flag for coco dataset. / default : true')
    parser.add_argument('--letter_box', action='store_true', help = ' If true, padding is performed to maintain the ratio of the input image. / default : false')
    parser.add_argument('--confidence_threshold', type=float, default=0.001)
    parser.add_argument('--iou_threshold', type=float, default=0.6)
    parser.add_argument('--score_threshold', type=float, default=0.001)

    args = parser.parse_args()
    args.mode = 'eval'
    args.batch_size = 1

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

    test_set = loader.DataLoader(args, hyp, 'eval')

    if args.only_coco_eval:
        map, map50 = coco_eval(args.json_path,test_set)
        pf = '%20s' + '%15.3g' * 2
        pc = '%20s' + '%15s' * 2
        print(pc % ('', 'coco mAP', 'coco mAP50'))
        print(pf % ('all',map, map50), '\n')
        print('coco eval')
    elif not args.is_coco:
        print('default test')
        mp,mr = test(args,hyp,test_set)
        pf = '%20s' + '%15.3g' * 2
        pc = '%20s' + '%15s' * 2
        
        print(pc % ('', 'mP', 'mR'))
        print(pf % ('all', mp, mr), '\n')
        print('coco eval')
    else:
        mp, mr, map50, map = test_coco(args,hyp,test_set)
        pf = '%20s' + '%15.3g' * 4
        pc = '%20s' + '%15s' * 4
        print(pc % ('','mP','mR','coco mAP','coco mAP50'))
        print(pf % ('all', mp, mr, map, map50), '\n')
        print('coco eval')