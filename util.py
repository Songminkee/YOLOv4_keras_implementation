import tensorflow as tf
import glob
import numpy as np
import math
import os

def mish(x,name):
    return x * tf.math.tanh(tf.math.softplus(x))

def upsample(x):
    return tf.image.resize(x, (x.shape[1] * 2, x.shape[2] * 2), method='bilinear')

def conv2d(x,filter,kernel,stride=1,name=None,activation='mish',gamma_zero=False):
    if stride==1:
        x = tf.keras.layers.Conv2D(filter,kernel,stride,padding='same',use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                               kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                               )(x)
    else:
        x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
        x = tf.keras.layers.Conv2D(filter, kernel, stride, padding='valid', use_bias=False,
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                   )(x)

    x = tf.keras.layers.BatchNormalization(momentum=0.9,epsilon=1e-5)(x)

    if activation=='mish':
        return mish(x,name)
    elif activation=='leaky':
        return tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        
def convset(x, filter):
    x = conv2d(x, filter, 1,activation='leaky')
    x = conv2d(x, filter * 2, 3,activation='leaky')
    x = conv2d(x, filter, 1,activation='leaky')
    x = conv2d(x, filter * 2, 3,activation='leaky')
    return conv2d(x, filter, 1,activation='leaky')


def load_class_name(data_root_path,classes_file):
    path = data_root_path+'/classes/'+classes_file
    classes = dict()
    with open(path,'r') as f:
        for label, name in enumerate(f):
            classes[label]=name.strip('\n')
    return classes

def load_coco_image_label_files(data_root_path,mode):
    image_txt_path = data_root_path+'/dataset/coco_{}2017.txt'.format(mode)
    images_path = [l.strip('\n') for l in open(image_txt_path,'r')]
    labels_path = ['/'+os.path.join(*im_path.split('/')[1:-3])+'/labels/{}2017/'.format(mode)+im_path.strip('jpg').split('/')[-1]+'txt' for im_path in images_path]
    return images_path,labels_path

def load_image_label_files(data_root_path,data_name,mode):
    image_txt_path = data_root_path+'/dataset/{}_{}.txt'.format(data_name,mode)
    images_path = [l.strip('\n') for l in open(image_txt_path,'r')]
    labels_path = ['/'+os.path.join(*im_path.split('/')[1:-3])+'/labels/{}/'.format(mode)+im_path.strip('jpg').split('/')[-1]+'txt' for im_path in images_path]
    return images_path,labels_path

def make_anchor(stride,anchor,is_tiny=False):
    if is_tiny:
        return np.reshape(anchor,(-1,3,2))/np.reshape(stride,(2,1,1))
    return np.reshape(anchor,(-1,3,2))/np.reshape(stride,(3,1,1))

def default_stride(is_tiny=False):
    if is_tiny:
        return np.array([16,32])
    return np.array([8,16,32])

def default_anchor(is_tiny=False):
    if is_tiny:
        return np.array([10, 14, 23,27, 37,58, 81,82, 135,169, 344,319])
    return np.array([12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401])

def default_sigmoid_scale(is_tiny=False):
    if is_tiny:
        return np.array([1.05,1.05])
    return np.array([1.2, 1.1, 1.05])

def box_iou(box1,box2):
    """
    Args:
        xywh format box
    """
    area1 = box1[..., 2] * box1[..., 3]
    area2 = box2[..., 2] * box2[..., 3]
    box1 = tf.concat([box1[..., :2] - box1[..., 2:] * 0.5,
                        box1[..., :2] + box1[..., 2:] * 0.5], axis=-1)
    box2 = tf.concat([box2[..., :2] - box2[..., 2:] * 0.5,
                        box2[..., :2] + box2[..., 2:] * 0.5], axis=-1)
    left_up = tf.maximum(box1[..., :2], box2[..., :2])
    right_down = tf.minimum(box1[..., 2:], box2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = area1 + area2 - inter_area

    return 1.0 * inter_area / union_area

def get_iou_for_test(pred_box,label):

    pred_box = tf.reshape(pred_box,[-1,1,4])
    pred_box = tf.tile(pred_box,[1,label.shape[0],1])
    label = tf.reshape(label,[1,-1,4])
    label = tf.tile(label, [pred_box.shape[0], 1, 1])

    p_x1, p_y1, p_x2, p_y2 = tf.split(pred_box, [1, 1, 1, 1], -1)
    l_x1, l_y1, l_x2, l_y2 = tf.split(label, [1, 1, 1, 1], -1)
    pw,ph,lw,lh = p_x2-p_x1,p_y2-p_y1,l_x2-l_x1,l_y2-l_y1

    con_x1 = tf.concat([p_x1, l_x1], -1)
    con_x2 = tf.concat([p_x2, l_x2], -1)
    con_y1 = tf.concat([p_y1, l_y1], -1)
    con_y2 = tf.concat([p_y2, l_y2], -1)

    inter = tf.expand_dims((tf.reduce_min(con_x2, -1) - tf.reduce_max(con_x1, -1)) * \
                           (tf.reduce_min(con_y2, -1) - tf.reduce_max(con_y1, -1)), -1)

    union = (pw * ph + 1e-16) + lw * lh - inter

    return tf.squeeze(inter / union)

def get_iou_loss(pred, label, method='GIoU'):
    nonan = tf.compat.v1.div_no_nan
    px, py, pw, ph = tf.split(pred, [1, 1, 1, 1], -1)
    lx, ly, lw, lh = tf.split(label, [1, 1, 1, 1], -1)

    p_x1, p_x2 = px - pw / 2.0, px + pw / 2.0
    p_y1, p_y2 = py - ph / 2.0, py + ph / 2.0
    l_x1, l_x2 = lx - lw / 2.0, lx + lw / 2.0
    l_y1, l_y2 = ly - lh / 2.0, ly + lh / 2.0

    in_left_up = tf.maximum(tf.concat([p_x1,p_y1],-1) , tf.concat([l_x1,l_y1],-1))
    in_right_down = tf.minimum(tf.concat([p_x2,p_y2],-1),tf.concat([l_x2,l_y2],-1))

    inter = tf.maximum(in_right_down - in_left_up, 0.0)
    inter = inter[..., 0] * inter[..., 1]
    
    union = tf.squeeze(pw * ph+ lw*lh,axis=-1) - inter
    iou = nonan(inter, union) * 1.0

    if method == 'IoU':
        return iou
    
    en_left_up = tf.minimum(tf.concat([p_x1,p_y1],-1) , tf.concat([l_x1,l_y1],-1))
    en_right_down = tf.maximum(tf.concat([p_x2,p_y2],-1),tf.concat([l_x2,l_y2],-1))
    enclose = tf.maximum(en_right_down - en_left_up, 0.0)

    if method == 'GIoU':
        c_area = enclose[...,0] * enclose[...,1]
        return iou - nonan((c_area - union), c_area)
    elif method == 'DIoU' or method == 'CIoU':
        c2 = enclose[...,0] *enclose[...,0] + enclose[...,1] *enclose[...,1]
        rho2 = tf.squeeze((lx - px) *(lx - px) + (ly - py) *(ly - py),axis=-1)
        diou_term = nonan(rho2,c2)
        if method == 'DIoU':
            return iou - diou_term
        else:
            atan_tum = tf.squeeze(tf.atan(nonan(pw, ph)) - tf.atan(nonan(lw, lh)),axis=-1)
            v = tf.stop_gradient(4 / (math.pi*math.pi) * atan_tum * atan_tum) #((tf.math.atan2(pw, ph) - tf.math.atan2(lw, lh)) ** 2)
            alpha = nonan(v, 1 - iou + v)
            return iou - nonan(rho2, c2) + v * alpha

# https://github.com/hunglc007/tensorflow-yolov4-tflite
def load_darknet_weights(model, weights_file, is_tiny=False,include_top=True,is_partial=False):
    if is_tiny:
        if is_partial:
            layer_size = 16
        elif include_top:
            layer_size = 21
        else:
            layer_size = 15
        output_pos = [17, 20]
    else:
        if is_partial:
            layer_size = 92
        elif include_top:
            layer_size = 110
        else:
            layer_size = 78
        output_pos = [93, 101, 109]
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)
    j = 0
    for i in range(layer_size):
        conv_layer_name = 'conv2d_%d' %i if i > 0 else 'conv2d'
        bn_layer_name = 'batch_normalization_%d' %j if j > 0 else 'batch_normalization'

        conv_layer = model.get_layer(conv_layer_name)
        filters = conv_layer.filters
        if i==output_pos and filters != 255:
            continue
        k_size = conv_layer.kernel_size[0]
        in_dim = conv_layer.input_shape[-1]

        if i not in output_pos:
            # darknet weights: [beta, gamma, mean, variance]
            bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
            # tf weights: [gamma, beta, mean, variance]
            bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
            bn_layer = model.get_layer(bn_layer_name)
            j += 1
        else:
            conv_bias = np.fromfile(wf,dtype=np.float32,count=filters)

        # darknet shape (out_dim, in_dim, height, width)
        conv_shape = (filters, in_dim, k_size, k_size)
        conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))

        # tf shape (height, width, in_dim, out_dim)
        conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])
        
        if i not in output_pos:
            conv_layer.set_weights([conv_weights])
            bn_layer.set_weights(bn_weights)
        else:
            conv_layer.set_weights([conv_weights, conv_bias])

    wf.close()

def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x

def merge_info(box,classes,stride,img_size=416):
    batch = tf.shape(box)[0]
    grid = img_size//stride

    xy,wh, conf, cls = tf.split(box, [2,2,1, classes], -1)
    cls_conf = conf * cls

    xywh = tf.concat([xy,wh],-1)*stride

    xywh = tf.reshape(xywh,[batch,3*(grid**2),4])
    cls_conf = tf.reshape(cls_conf,[batch,3*(grid**2),classes])
    return tf.concat([xywh, cls_conf],-1)

def get_decoded_pred(YOLO):
    '''
    The result is the same as applying the decode function after YOLO.pred.
        :param yolo: Yolo model class
        :return: xywh, cls
    '''
    feat = YOLO.box_feature
    cls = []
    xywh = []
    for i, box in enumerate(feat):
        grid = YOLO.img_size // YOLO.stride[i]
        conv_raw_dxdy_0, conv_raw_dwdh_0, conv_raw_score_0, \
        conv_raw_dxdy_1, conv_raw_dwdh_1, conv_raw_score_1, \
        conv_raw_dxdy_2, conv_raw_dwdh_2, conv_raw_score_2 = tf.split(box,
                                                                      (2, 2, 1 + YOLO.num_classes, 2, 2,
                                                                       1 + YOLO.num_classes,
                                                                       2, 2, 1 + YOLO.num_classes), axis=-1)
        conv_raw_score = [conv_raw_score_0, conv_raw_score_1, conv_raw_score_2]
        for idx, score in enumerate(conv_raw_score):
            score = tf.sigmoid(score)
            score = score[:, :, :, 0:1] * score[:, :, :, 1:]
            conv_raw_score[idx] = tf.reshape(score, (1, -1, YOLO.num_classes))
        pred_prob = tf.concat(conv_raw_score, axis=1)
        conv_raw_dwdh = [conv_raw_dwdh_0, conv_raw_dwdh_1, conv_raw_dwdh_2]

        for idx, dwdh in enumerate(conv_raw_dwdh):
            dwdh = tf.exp(dwdh) * YOLO.anchors[i][idx]
            conv_raw_dwdh[idx] = tf.reshape(dwdh, (1, -1, 2))
        pred_wh = tf.concat(conv_raw_dwdh, axis=1)

        xy_grid = tf.meshgrid(tf.range(grid), tf.range(grid))
        xy_grid = tf.stack(xy_grid, axis=-1)  # [gx, gy, 2]
        xy_grid = tf.expand_dims(xy_grid, axis=0)
        xy_grid = tf.cast(xy_grid, tf.float32)

        conv_raw_dxdy = [conv_raw_dxdy_0, conv_raw_dxdy_1, conv_raw_dxdy_2]
        for idx, dxdy in enumerate(conv_raw_dxdy):
            dxdy = ((tf.sigmoid(dxdy) * YOLO.sigmoid_scale[i]) - 0.5 * (YOLO.sigmoid_scale[i] - 1) + xy_grid)
            conv_raw_dxdy[idx] = tf.reshape(dxdy, (1, -1, 2))
        pred_xy = tf.concat(conv_raw_dxdy, axis=1)
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1) * YOLO.stride[i]
        cls.append(pred_prob)
        xywh.append(pred_xywh)
    cls = tf.concat(cls, 1)
    xywh = tf.concat(xywh, 1)
    return (xywh, cls)

def decode(yolo,input):
    '''
        :param yolo: Yolo model class
        :param input: Input image
        :param args: args must have information about confidence_threshold, img_size.
        :return: box,cls_conf
    '''
    boxes = yolo.model(input, training=False)
    boxes = tf.concat([merge_info(box, yolo.num_classes, yolo.stride[i],yolo.img_size) for i, box in enumerate(boxes)], 1)

    # Eliminate low confidence
    xywh, cls = tf.split(boxes, [4, yolo.num_classes], -1)
    return xywh,cls

def tf_nms_format(xywh,cls,args):
    '''
        :param xywh,cls_conf: decoded information. xywh =bbox.
        :param args: args must have information about batch_size, iou_threshold, score_threshold.
        :return: boxes, scores, classes, valid_detections information through NMS
    '''
    scores_max = tf.math.reduce_max(cls, axis=-1)
    mask = scores_max >= args.confidence_threshold

    class_boxes = tf.boolean_mask(xywh, mask)
    pred_conf = tf.boolean_mask(cls, mask)
    class_boxes = tf.reshape(class_boxes, [tf.shape(cls)[0], -1, tf.shape(class_boxes)[-1]])
    cls_conf = tf.reshape(pred_conf, [tf.shape(cls)[0], -1, tf.shape(pred_conf)[-1]])

    # Convert to tf_nms format
    box_xy, box_wh = tf.split(class_boxes, (2, 2), axis=-1)
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    box_mins = (box_yx - (box_hw / 2.)) / args.img_size
    box_maxes = (box_yx + (box_hw / 2.)) / args.img_size
    xyxy = tf.concat([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ], axis=-1)
    return xyxy,cls_conf

def tf_nms(xyxy,cls_conf,args):
    '''
        :param box,cls_conf: decoded information. xyxy = bbox.
        :param args: args must have information about batch_size, iou_threshold, score_threshold.
        :return: boxes, scores, classes, valid_detections information through NMS
    '''
    # NMS
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(xyxy, (args.batch_size, -1, 1, 4)),
        scores=tf.reshape(
            cls_conf, (args.batch_size, -1, tf.shape(cls_conf)[-1])),
        max_output_size_per_class=100,
        max_total_size=100,
        iou_threshold=args.iou_threshold,
        score_threshold=args.score_threshold,
    )
    return boxes, scores, classes, valid_detections

def inference(xywh,cls,args):
    '''
    :param xywh,cls: decoded information by tf_lite_pred or decode function.
    :param input: Input image
    :param args: args must have information about confidence_threshold, img_size,batch_size,iou_threshold, score_threshold.
    :return: boxes, scores, classes, valid_detections information through NMS
    '''
    xyxy,cls_conf = tf_nms_format(xywh,cls,args)
    return tf_nms(xyxy,cls_conf,args)


def convert_to_origin_shape(box,pad=None,ratio=None,h0=None,w0=None,h=None,w=None,letter_box=False):
    '''
    :return: Convert the box information to information about the original original image.
    '''
    y_min, x_min, y_max, x_max = tf.split(box,[1,1,1,1],-1)
    if letter_box:
        left = int(round(pad[0] - 0.1))
        top = int(round(pad[1] - 0.1))
        x_min = (x_min * w - left) / ratio[0] / (w - pad[0] * 2) * w0
        y_min = (y_min * h - top) / ratio[1] / (h - pad[1] * 2) * h0
        x_max = (x_max * w - left) / ratio[0] / (w - pad[0] * 2) * w0
        y_max = (y_max * h - top) / ratio[1] / (h - pad[1] * 2) * h0
    else:
        x_min *= w0
        y_min *= h0
        x_max *= w0
        y_max *= h0
    return y_min,x_min,y_max,x_max


def scaled_xywh2xyxy(box,h,w):
    xyxy = np.zeros_like(box)
    xyxy[:, 0] = (box[:, 0] - box[:, 2] / 2) * w
    xyxy[:, 1] = (box[:, 1] - box[:, 3] / 2) * h
    xyxy[:, 2] = (box[:, 0] + box[:, 2] / 2) * w
    xyxy[:, 3] = (box[:, 1] + box[:, 3] / 2) * h
    return xyxy

def ap_per_class(tp,conf,pred_cls,label_cls):
    i = np.argsort(-conf)

    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    unique_cls = np.unique(label_cls)
    ap,p,r = np.zeros(len(unique_cls)) , np.zeros(len(unique_cls)), np.zeros(len(unique_cls))
    for ci, c in enumerate(unique_cls):
        i = pred_cls == c
        n_gt = np.sum(label_cls==c) # tp+nf, 실제 정답의 개수
        n_p = np.sum(i) # tp

        if not n_gt or not n_p:
            continue
        # Accumulate FPs and TPs
        fpc = np.cumsum(1-tp[i])
        tpc = np.cumsum(tp[i])

        # Recall
        recall = tpc / (n_gt + 1e-16)  # recall curve
        r[ci] = np.interp(-0.1, -conf[i], recall)  # r at pr_score, negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)
        p[ci] = np.interp(-0.1, -conf[i], precision)

        mrec = np.concatenate(([0.], recall, [min(recall[-1] + 1E-3, 1.)]))
        mpre = np.concatenate(([0.], precision, [0.]))
        
        # Compute the precision envelope
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap[ci] = np.trapz(np.interp(x, mrec, mpre), x)  # integrate

    # Compute F1 score (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)
    return p, r, ap, f1, unique_cls.astype('int32')

def get_xywh_loss(pred_xywh,true_xywh,mask,anchors):
    pred_xy, pred_wh = tf.split(pred_xywh,[2,2],-1)
    true_xy, true_wh = tf.split(true_xywh,[2,2],-1)
    
    pred_xy = pred_xy - tf.floor(true_xy)
    true_xy = true_xy - tf.floor(true_xy)
    xy_loss = tf.reduce_sum(tf.clip_by_value(tf.where(mask,tf.square(pred_xy-true_xy),0.),0.,1e3))
    
    pred_wh = tf.math.log(tf.clip_by_value(tf.math.divide_no_nan(pred_wh,anchors),1e-10,1e3))
    pred_wh = tf.where(tf.math.is_inf(pred_wh),
                           tf.zeros_like(pred_wh), pred_wh) 
    true_wh = tf.math.log(tf.clip_by_value(tf.math.divide_no_nan(true_wh,anchors),1e-10,1e3))
    true_wh = tf.where(tf.math.is_inf(true_wh),
                           tf.zeros_like(true_wh), true_wh)    
    wh_loss = tf.reduce_sum(tf.clip_by_value(tf.where(mask,tf.square(pred_wh-true_wh),0.),0.,1e3))
    return xy_loss + wh_loss


def freeze_layer(YOLO,args):
    if args.partial:
        print('partial weight')
        if args.is_tiny:
            num=16
        else:
            num=92
        
        for layer in YOLO.model.layers:
            try:
                layer_num =  int(layer.name.split('_')[-1])
            except:
                layer_num = 0

            if ('batch_normalization' in layer.name or 'conv2d' in layer.name) and num<layer_num:
                layer.trainable=True
            else:
                layer.trainable=False
    elif args.freeze_conv:
        print('freeze_conv')
        for layer in YOLO.backbone.layers:
            layer.trainable = False
    else:
        pass