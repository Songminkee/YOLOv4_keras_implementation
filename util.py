import tensorflow as tf
import glob
import numpy as np
import math
import os

def mish(x,name):
    return x * tf.nn.tanh( tf.nn.softplus(x),name=name)

def upsample(x):
    return tf.image.resize(x, (x.shape[1] * 2, x.shape[2] * 2), method='bilinear')

def conv2d(x,filter,kernel,stride=1,name=None,activation='mish',gamma_zero=False):
    if stride==1:
        x = tf.keras.layers.Conv2D(filter,kernel,stride,padding='same',use_bias=False,
                               kernel_initializer=tf.random_normal_initializer(stddev=0.01))(x)
    else:
        x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
        x = tf.keras.layers.Conv2D(filter, kernel, stride, padding='valid', use_bias=False,
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.01))(x)

    if gamma_zero:
        x = tf.keras.layers.BatchNormalization(momentum=0.03, epsilon=1e-4,gamma_initializer='zeros')(x)
    else:
        x = tf.keras.layers.BatchNormalization( momentum=0.03, epsilon=1e-4)(x)

    if activation=='mish':
        return mish(x,name)
    elif activation=='leaky':
        return tf.nn.leaky_relu(x, alpha=0.1)#tf.keras.layers.LeakyReLU(name=name,alpha=0.1)(x)

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
        return np.array([23,27, 37,58, 81,82, 81,82, 135,169, 344,319])
    return np.array([12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401])

def default_sigmoid_scale(is_tiny=False):
    if is_tiny:
        return np.array([1.05,1.05])
    return np.array([1.2, 1.1, 1.05])

def wh_iou(anchor, label):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    # wh1 = wh1[:, None]  # [N,1,2]
    # wh2 = wh2[None]  # [1,M,2]
    b,max_label,_=label.shape # [batch,max_label,2]

    anchor = tf.reshape(tf.cast(anchor,tf.float32),[1,1,3,2])
    anchor = tf.tile(anchor,[b,max_label,1,1]) # [batch,3,max_label,2]
    label = tf.tile(tf.expand_dims(label, 2),[1,1,3,1]) # [batch,3,max_label,2]

    anchor_w,anchor_h = tf.split(anchor,([1,1]),-1)
    label_w,label_h = tf.split(label,([1,1]),-1)

    min_w = tf.reduce_min(tf.concat([anchor_w,label_w],-1),keepdims=True,axis=-1)
    min_h = tf.reduce_min(tf.concat([anchor_h,label_h],-1),keepdims=True,axis=-1)
    inter = min_w * min_h

    return inter / (label_w*label_h + anchor_h*anchor_w - inter)  # iou = inter / (area1 + area2 - inter)

def get_iou(pred_box,label):

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

def label_scaler(out):
    b,h,w,_,_ = out.shape
    return tf.cast(tf.reshape([1,w,h,w,h],[1,1,5]),tf.float32)

def get_idx(label):
    b,max_label,_ = label.shape
    _,gx,gy,_ = tf.split(label,[1,1,1,2],-1)
    return tf.concat([tf.tile(tf.reshape(tf.range(0,b),[-1,1,1]),[1,max_label,1]),tf.cast(gy,tf.int32),tf.cast(gx,tf.int32)],-1)

def build_overlap_target(anchor, label,hyp,out,is_best):
    '''
    Build target with overlapped coordinate
    out : [b,grid,grid,3,(5+num_classes)]
    is_best : [b,max_label,3,1] -> [b,max_label*3,5]
    anchor : [3,2]
    label : [b,max_label,5] -> [b,max_label*3,5]
    '''
    _, max_y, max_x, _, _ = out.shape
    c, x, y, wh = tf.split(label, [1, 1, 1, 2], -1)
    x_n = tf.where(tf.math.mod(x, 1.) > 0.5, x + 1, x - 1)
    x_n = tf.where(tf.logical_and(x_n >= 0, x_n < max_x), tf.concat([c, x_n, y, wh], -1), 0)

    y_n = tf.where(tf.math.mod(y, 1.) > 0.5, y + 1, y - 1)
    y_n = tf.where(tf.logical_and(y_n >= 0, y_n < max_y), tf.concat([c, x, y_n, wh], -1), 0)

    label = tf.concat([label, x_n, y_n], 1)
    return build_target(anchor,label,hyp,tf.tile(is_best,[1,3,1,1]))


def build_target(anchor,label,hyp,is_best):
    '''
    is_best : [b,max_label,3,1]
    anchor : [3,2]
    label : [b,max_label,5]
    '''
    iou = wh_iou(anchor, label[..., 3:])  # [b,max_label,3,1] 각 anchor 별 label과 iou
    idx = get_idx(label)
    label = tf.tile(tf.expand_dims(label,2),[1,1,3,1])
    is_label = tf.reduce_sum(label, -1, keepdims=True) != 0
    mask = tf.logical_and(is_label,tf.logical_or(iou> hyp['iou_t'],is_best))
    return mask,idx,label


def get_iou_loss(pred, label, method='GIoU'):
    nonan = tf.compat.v1.div_no_nan
    px, py, pw, ph = tf.split(pred, [1, 1, 1, 1], -1)
    lx, ly, lw, lh = tf.split(label, [1, 1, 1, 1], -1)

    p_x1, p_x2 = px - pw / 2.0, px + pw / 2.0
    p_y1, p_y2 = py - ph / 2.0, py + ph / 2.0
    l_x1, l_x2 = lx - lw / 2.0, lx + lw / 2.0
    l_y1, l_y2 = ly - lh / 2.0, ly + lh / 2.0

    con_x1 = tf.concat([p_x1, l_x1], -1)
    con_x2 = tf.concat([p_x2, l_x2], -1)
    con_y1 = tf.concat([p_y1, l_y1], -1)
    con_y2 = tf.concat([p_y2, l_y2], -1)

    max_x1 = tf.reduce_max(con_x1, -1)
    min_x2 = tf.reduce_min(con_x2, -1)

    max_y1 = tf.reduce_max(con_y1, -1)
    min_y2 = tf.reduce_min(con_y2, -1)

    inter = tf.expand_dims((min_x2 - max_x1) * (min_y2 - max_y1), -1)

    union = pw * ph + lw * lh - inter
    iou = nonan(inter, union)

    # where non overlapped
    iou = tf.where(tf.expand_dims(tf.logical_and(min_y2 > max_y1, min_x2 > max_x1), -1), iou, 0.)

    if method == 'IoU':
        return iou

    cw = tf.reduce_max(con_x2, -1, keepdims=True) - tf.reduce_min(con_x1, -1, keepdims=True)
    ch = tf.reduce_max(con_y2, -1, keepdims=True) - tf.reduce_min(con_y1, -1, keepdims=True)

    if method == 'GIoU':
        c_area = cw * ch
        return iou - nonan((c_area - union), c_area)
    elif method == 'DIoU' or method == 'CIoU':
        c2 = cw ** 2 + ch ** 2
        rho2 = (lx - px) ** 2 + (ly - py) ** 2
        if method == 'DIoU':
            return iou - nonan(rho2, c2)
        else:
            v = (4 / ((math.pi) ** 2)) * ((tf.math.atan2(pw, ph) - tf.math.atan2(lw, lh)) ** 2)
            alpha = nonan(v, 1 - iou + v)
            return iou - (nonan(rho2, c2) + v * alpha)

def smoothing_value(classes,eps=0.0):
    return (1.0-eps),eps/classes

def get_threshold_mask(pred,label,ignore_threshold=0.7):
    pred = tf.tile(tf.expand_dims(pred, -2), [1, 1, 1, 1, tf.shape(label)[1], 1])
    label = tf.tile(tf.reshape(label, [tf.shape(pred)[0], 1, 1, 1, -1, 4]),
                    [1, tf.shape(pred)[1], tf.shape(pred)[1], 3, 1, 1])
    ious = tf.clip_by_value(get_iou_loss(pred, label, 'IoU'), 0.0, 1.0)
    best_iou = tf.reduce_max(tf.where(tf.reduce_sum(label, axis=-1, keepdims=True) > 0., ious, 0.), 4)
    return tf.squeeze(best_iou > ignore_threshold), best_iou

def get_best_match_anchor(label,anchors):
    ious = [wh_iou(anchors[i], label[i][..., 3:]) for i in range(len(anchors))]
    ious = tf.concat(ious, 2) # [b, max_label, anchors*3, 1]
    best = tf.logical_and(ious == tf.reduce_max(ious,2,keepdims=True),ious>0.)
    return list(tf.split(best,[len(anchors) for _ in range(len(anchors))],axis=2)) # [b, max_label, 3, 1] x anchors

# https://github.com/hunglc007/tensorflow-yolov4-tflite
def load_darknet_weights(model, weights_file, is_tiny=False,include_top=True):
    if is_tiny:
        if include_top:
            layer_size = 21
        else:
            layer_size = 15
        output_pos = [17, 20]
    else:
        if include_top:
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
            try:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
            except:
                pass


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

def merge_info(box,classes,stride):
    batch = tf.shape(box)[0]
    if classes>1:
        xy,wh, conf, cls = tf.split(box, [2,2,1, classes], -1)
        cls_conf = conf * cls
    else:
        xy, wh, cls_conf = tf.split(box, [2, 2, 1], -1)
    xywh = tf.concat([xy,wh],-1)*stride

    xywh = tf.reshape(xywh,[batch,-1,4])
    cls_conf = tf.reshape(cls_conf,[batch,-1,classes])
    return tf.concat([xywh, cls_conf],-1)

def inference(yolo,input,args):
    '''
    :param yolo: Yolo model class
    :param input: Input image
    :param args: args must have information about confidence_threshold, img_size,batch_size,iou_threshold, score_threshold.
    :return: boxes, scores, classes, valid_detections information through NMS
    '''
    boxes = yolo.model(input,training=False)
    boxes = tf.concat([merge_info(box, yolo.num_classes, yolo.stride[i]) for i, box in enumerate(boxes)], 1)

    # Eliminate low confidence
    xywh, cls = tf.split(boxes, [4, yolo.num_classes], -1)

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
    box = tf.concat([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ], axis=-1)

    # NMS
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(box, (args.batch_size, -1, 1, 4)),
        scores=tf.reshape(
            cls_conf, (args.batch_size, -1, tf.shape(cls_conf)[-1])),
        max_output_size_per_class=100,
        max_total_size=100,
        iou_threshold=args.iou_threshold,
        score_threshold=args.score_threshold,
    )
    return boxes, scores, classes, valid_detections

def convert_to_origin_shape(box,pad=None,ratio=None,h0=None,w0=None,h=None,w=None,is_padding=False):
    '''
    :return: Convert the box information to information about the original original image.
    '''
    y_min, x_min, y_max, x_max = tf.split(box,[1,1,1,1],-1)
    if is_padding:
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
        n_gt = np.sum(label_cls==c)
        n_p = np.sum(i)

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