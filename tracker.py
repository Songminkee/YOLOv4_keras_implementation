from timeit import time
import warnings
import cv2
import numpy as np
import argparse

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet
from util import convert_to_origin_shape, load_class_name
from detect import model_detection, select_yolo
from collections import deque
import tensorflow as tf


pts = [deque(maxlen=30) for _ in range(9999)]
warnings.filterwarnings('ignore')

# initialize a list of colors to represent each possible class label
np.random.seed(100)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

def main(args, hyp):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    counter = []
    # deep_sort
    model_filename = 'saved_model/market1501'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    # select yolo
    YOLO, input_details, output_details,saved_model_loaded = select_yolo(args, hyp)

    video_capture = cv2.VideoCapture(args.video)

    #Define the codec and create VideoWriter object
    if args.write_video:
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('./output/output.avi', fourcc, 15, (w, h))
        list_file = open('detection_rslt.txt', 'w')
        frame_index = -1

    fps = 0.0
    convert_class_name = load_class_name(args.data_root, args.class_file)
    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if not ret:
            break
        t1 = time.time()

        image = np.squeeze(frame)
        img = cv2.cvtColor(image.copy(),cv2.COLOR_BGR2RGB) / 255.0
        h, w, _ = img.shape
        if h != args.img_size or w != args.img_size:
            img = cv2.resize(img, (args.img_size, args.img_size))
        inf_time = time.time()
        
        boxes, confidence, class_names, valid_detections= model_detection(img, YOLO, args, input_details, output_details)
        print("inf time",time.time()-inf_time)
        tran_time = time.time()
        y_min, x_min, y_max, x_max = convert_to_origin_shape(boxes, None, None, h, w)
        w,h = x_max-x_min , y_max-y_min
        boxes = np.concatenate([x_min, y_min, w, h], -1)

        boxes = tf.squeeze(boxes, 0)      # 100, 4
        class_names = tf.squeeze(class_names, 0)
        print("tran time",time.time()-tran_time)        
        st_t = time.time()
        features = encoder(frame, boxes[:valid_detections[0]])
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxes, features)]
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        print("tracker time",time.time()-st_t)
        i = int(0)
        indexIDs = []

        draw_st = time.time()
        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            indexIDs.append(int(track.track_id))
            counter.append(int(track.track_id))
            bbox = track.to_tlbr()
            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(color), 3)
            cv2.putText(frame,str(track.track_id),(int(bbox[0]), int(bbox[1] -50)),0, 5e-3 * 150, (color),2)
            if len(class_names) > 0:
               cls = np.int8(class_names[0])
               cv2.putText(frame, str(convert_class_name[cls]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (color),2)

            i += 1
            center = (int(((bbox[0])+(bbox[2]))/2),int(((bbox[1])+(bbox[3]))/2))

            pts[track.track_id].append(center)

            thickness = 5
            # center point
            cv2.circle(frame,  (center), 1, color, thickness)

            # draw motion path
            for j in range(1, len(pts[track.track_id])):
                if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                    continue
                thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                cv2.line(frame,(pts[track.track_id][j-1]), (pts[track.track_id][j]),(color),thickness)
            if args.write_video:
                list_file.write(str(frame_index) + ',')
                list_file.write(str(track.track_id) + ',')
                b0 = str(bbox[0])  # .split('.')[0] + '.' + str(bbox[0]).split('.')[0][:1]
                b1 = str(bbox[1])  # .split('.')[0] + '.' + str(bbox[1]).split('.')[0][:1]
                b2 = str(bbox[2] - bbox[0])  # .split('.')[0] + '.' + str(bbox[3]).split('.')[0][:1]
                b3 = str(bbox[3] - bbox[1])

                list_file.write(str(b0) + ',' + str(b1) + ',' + str(b2) + ',' + str(b3) + '\n')

        cv2.putText(frame, "Current Box Counter: "+str(i),(int(20), int(80)),0, 5e-3 * 200, (0,255,0),2)
        cv2.putText(frame, "FPS: %f"%(fps),(int(20), int(40)),0, 5e-3 * 200, (0,255,0),3)
        cv2.namedWindow("YOLO4_Deep_SORT", 0)
        cv2.resizeWindow('YOLO4_Deep_SORT', 1024, 768)
        cv2.imshow('YOLO4_Deep_SORT', frame)

        if args.write_video:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1

        fps = (fps + (1./(time.time()-t1))) / 2
        print("draw_time",time.time()-draw_st)
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='YOLOv4 Test')
    parser.add_argument('--video', help="path to input video", default="./test_video/TownCentreXVID.avi")
    parser.add_argument('--img_size',              type=int,   help='Size of input image / default : 416', default=416)
    parser.add_argument('--data_root',              type=str,   help='Root path of class name file and coco_%2017.txt / default : "./data"', default='./data')
    parser.add_argument('--class_file',              type=str,   help='Class name file / default : "coco.name"', default='box.names') # 'coco.names'
    parser.add_argument('--num_classes', type=int, help='Number of classes (in COCO 80) ', default=80) # 80
    parser.add_argument('--weight_path',type=str,default='weight/box/final', help='path of weight')     # 'dark_weight/yolov4.weights'
    parser.add_argument('--is_saved_model', action='store_true',help = 'If ture, load saved model')
    parser.add_argument('--is_tflite', action='store_true', help='If ture, load saved model')
    parser.add_argument('--is_darknet_weight', action='store_true', help = 'If true, load the weight file used by the darknet framework.') # 'store_false'
    parser.add_argument('--is_tiny', action='store_true', help = 'Flag for using tiny. / default : false')
    parser.add_argument('--write_video', action='store_true', help='Flag for save result video')
    parser.add_argument('--confidence_threshold', type=float, default=0.001)
    parser.add_argument('--iou_threshold', type=float, default=0.1)
    parser.add_argument('--score_threshold', type=float, default=0.1)
    parser.add_argument('--data_name', type=str,
                        help='Root path of class name file and coco_%2017.txt / default : "./data"'
                        , default='coco')
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

    main(args, hyp)
