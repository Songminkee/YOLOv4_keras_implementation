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
    x = tf.keras.layers.Conv2D(filter,kernel,stride,padding='same',use_bias=False,
                               kernel_initializer=tf.random_normal_initializer(stddev=0.01))(x)
    if gamma_zero:
        x = tf.keras.layers.BatchNormalization(momentum=0.03, epsilon=1e-4,gamma_initializer='zeros')(x)
    else:
        x = tf.keras.layers.BatchNormalization( momentum=0.03, epsilon=1e-4)(x)

    if activation=='mish':
        return mish(x,name)
    elif activation=='leaky':
        return tf.keras.layers.LeakyReLU(name=name)(x)

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

def label_scaler(out):
    b,h,w,_,_ = out.shape
    return tf.cast(tf.reshape([1,w,h,w,h],[1,1,5]),tf.float32)

def get_idx(label):
    b,max_label,_ = label.shape
    _,gx,gy,_ = tf.split(label,[1,1,1,2],-1)
    return tf.concat([tf.tile(tf.reshape(tf.range(0,b),[-1,1,1]),[1,max_label,1]),tf.cast(tf.floor(gy),tf.int32),tf.cast(tf.floor(gx),tf.int32)],-1)

def build_target(anchor,label,hyp):
    iou = wh_iou(anchor, label[..., 3:])  # [b,max_label,3,1] 각 anchor 별 label과 iou
    mask = iou > hyp['iou_t']
    idx = get_idx(label)
    label = tf.tile(tf.expand_dims(label,2),[1,1,3,1])
    return tf.stop_gradient(mask),tf.stop_gradient(idx),tf.stop_gradient(label)

def get_iou_loss(pred,label,method='GIoU'):
    px, py, pw, ph = tf.split(pred, [1, 1, 1, 1], -1)
    lx, ly, lw, lh = tf.split(label, [1, 1, 1, 1], -1)

    p_x1, p_x2 = px - pw / 2.0, px + pw / 2.0
    p_y1, p_y2 = py - ph / 2.0, py + ph / 2.0
    l_x1, l_x2 = lx - lw / 2.0, lx + lw / 2.0
    l_y1, l_y2 = lw - lh / 2.0, lw + lh / 2.0

    con_x1 = tf.concat([p_x1, l_x1], -1)
    con_x2 = tf.concat([p_x2, l_x2], -1)
    con_y1 = tf.concat([p_y1, l_y1], -1)
    con_y2 = tf.concat([p_y2, l_y2], -1)

    inter = tf.expand_dims((tf.reduce_min(con_x2,-1) - tf.reduce_min(con_x1,-1)) * \
            (tf.reduce_min(con_y2, -1) - tf.reduce_min(con_y1, -1)),-1)

    union = (pw * ph + 1e-16) + lw*lh - inter
    iou = inter/union

    cw = tf.reduce_max(con_x2,-1,keepdims=True) - tf.reduce_min(con_x1,-1,keepdims=True)
    ch = tf.reduce_max(con_y2,-1,keepdims=True) - tf.reduce_min(con_y1,-1,keepdims=True)

    if method=='GIoU':
        c_area = cw * ch + 1e-16
        return iou - (c_area - union) / c_area
    elif method=='DIoU' or method=='CIoU':
        c2 = cw**2 + ch**2 +1e-16
        rho2 = ((l_x1+l_x2) - (p_x1 + p_x2)) ** 2 / 4 + ((l_y1+l_y2) - (p_y1+p_y2)) ** 2 / 4
        if method=='DIoU':
            return iou-rho2/c2
        else:
            v = (4 / math.pi**2) * (tf.math.atan((l_x2-l_x1)/(l_y2-l_y1)) - tf.math.atan((p_x2-p_x1)/(p_y2-p_y1))**2)
            alpha = tf.stop_gradient(iou / (1-iou+v))
            return iou - (rho2 / c2 + v * alpha)
    else:
        return iou

def smoothing_value(classes,eps=0.0):
    return (1.0-eps),eps/classes

# https://github.com/hunglc007/tensorflow-yolov4-tflite
def load_darknet_weights(model, weights_file, is_tiny=False):
    if is_tiny:
        layer_size = 21
        output_pos = [17, 20]
    else:
        layer_size = 110
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
            conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

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
