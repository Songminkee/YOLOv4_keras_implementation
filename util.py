import tensorflow as tf
import glob
import numpy as np

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
    images_path = ['.'+l.strip('\n') for l in open(image_txt_path,'r')]
    labels_path = [data_root_path+'/dataset/COCO/labels/{}2017/'.format(mode)+im_path.strip('jpg').split('/')[-1]+'txt' for im_path in images_path]
    return images_path,labels_path

def make_anchor(stride,anchor):
    return np.reshape(anchor,(-1,3,2))/np.reshape(stride,(3,1,1))

# print(load_class_name('./data','coco.names'))
#
# print(load_coco_image_label_files('./data','val'))