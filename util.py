import tensorflow as tf
import glob
import numpy as np
import math

def mish(x,name):
    return x * tf.nn.tanh( tf.nn.softplus(x),name=name)

def upsample(x):
    return tf.image.resize(x, (x.shape[1] * 2, x.shape[2] * 2), method='bilinear')

def conv2d(x,filter,kernel,stride=1,name=None,activation='mish'):
    x = tf.keras.layers.Conv2D(filter,kernel,stride,padding='same',use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
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
    labels_path = [data_root_path+'/dataset/COCO/labels/{}2017/'.format(mode)+im_path.strip('jpg').split('/')[-1]+'txt' for im_path in images_path]

    return images_path,labels_path

def make_anchor(stride,anchor):
    return np.reshape(anchor,(-1,3,2))/np.reshape(stride,(3,1,1))

def default_stride():
    return np.array([8,16,32])

def default_anchor():
    return np.array([12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401])

def default_sigmoid_scale():
    return np.array([1.2, 1.1, 1.05])

def wh_iou(anchor, label):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    # wh1 = wh1[:, None]  # [N,1,2]
    # wh2 = wh2[None]  # [1,M,2]
    b,max_label,_=label.shape # [batch,max_label,2]
    #print("origin_label",label)
    anchor = tf.reshape(tf.cast(anchor,tf.float32),[1,1,3,2])
    anchor = tf.tile(anchor,[b,max_label,1,1]) # [batch,3,max_label,2]
    label = tf.tile(tf.expand_dims(label, 2),[1,1,3,1]) # [batch,3,max_label,2]

    anchor_w,anchor_h = tf.split(anchor,([1,1]),-1)
    label_w,label_h = tf.split(label,([1,1]),-1)

    min_w = tf.reduce_min(tf.concat([anchor_w,label_w],-1),keepdims=True,axis=-1)
    min_h = tf.reduce_min(tf.concat([anchor_h,label_h],-1),keepdims=True,axis=-1)
    inter = min_w*min_h
    # print("inter",inter)
    # print("label",label_h*label_w)
    # print("an",anchor_h * anchor_w)
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
    mask = iou>hyp['iou_t']
    idx = get_idx(label)
    label = tf.tile(tf.expand_dims(label,2),[1,1,3,1])
    return mask,idx,label

def get_iou_loss(pred,label,method='GIoU'):
    px, py, pw, ph = tf.split(pred, [1, 1, 1, 1], -1)
    lx, ly, lw, lh = tf.split(label, [1, 1, 1, 1], -1)

    p_x1, p_x2 = px - pw / 2, px + pw / 2
    p_y1, p_y2 = py - ph / 2, py + ph / 2
    l_x1, l_x2 = lx - lw / 2, lx + lw / 2
    l_y1, l_y2 = lw - lh / 2, lw + lh / 2

    con_x1 = tf.concat([p_x1, l_x1], -1)
    con_x2 = tf.concat([p_x2, l_x2], -1)
    con_y1 = tf.concat([p_y1, l_y1], -1)
    con_y2 = tf.concat([p_y2, l_y2], -1)

    inter = tf.expand_dims((tf.reduce_min(con_x2,-1) - tf.reduce_min(con_x1,-1)) * \
            (tf.reduce_min(con_y2, -1) - tf.reduce_min(con_y1, -1)),-1)

    union = (pw*ph + 1e-16) + lw*lh - inter
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

# print(load_class_name('./data','coco.names'))
#
# print(load_coco_image_label_files('./data','val'))