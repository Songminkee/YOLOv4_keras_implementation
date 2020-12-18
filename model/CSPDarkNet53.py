import tensorflow as tf
from util import *
import argparse

def resblock(input,filter,i,j):
    x = conv2d(input,filter//2,1,name='block_{}_{}_conv1'.format(i,j))
    if i:
        x = conv2d(x,filter//2,3,name='block_{}_{}_conv3'.format(i,j),gamma_zero=True)
    else:
        x = conv2d(x,filter,3,name='block_{}_{}_conv3'.format(i,j),gamma_zero=True)

    return input+x

class CSPDarkNet53(object):
    def __init__(self,args,num_classes=None,include_top=False):
        self.batch_size = args.batch_size
        self.input_shape = [args.img_size,args.img_size,3]
        self.filter = [64,128,256,512,1024]
        self.res_iter = [1,2,8,8,4]
        self.include_top = include_top
        self.num_classes = num_classes
        self.model = self.build()

    def build(self):
        model = self.backbone()
        if not self.include_top:
            return model
        return tf.keras.Model(inputs=model.input,outputs=self.head(model.output))

    def backbone(self):
        input = tf.keras.layers.Input(shape=(self.input_shape[0], self.input_shape[1], 3))
        x = conv2d(input, 32, 3)
        out = []
        for i, iter in enumerate(self.res_iter):
            x = conv2d(x, self.filter[i], 3, 2, name='block_{}_down'.format(i))
            if not i:
                route = conv2d(x, self.filter[i], 1, name='block_{}_down'.format(i))
                x = conv2d(x, self.filter[i], 1, name='block_{}_conv1'.format(i))
            else:
                route = conv2d(x, self.filter[i]//2, 1, name='block_{}_down'.format(i))
                x = conv2d(x, self.filter[i]//2, 1, name='block_{}_conv1'.format(i))

            for j in range(iter):
                x = resblock(x, self.filter[i], i, j)

            if not i:
                x = conv2d(x, self.filter[i], 1, name='block_{}_conv1_2'.format(i))
                x = tf.concat([x,route],axis=-1,name='concat_{}'.format(i))
                x = conv2d(x,self.filter[i],1,name='block_{}_out'.format(i))
            else:
                x = conv2d(x, self.filter[i]//2, 1, name='block_{}_conv1_2'.format(i))
                x = tf.concat([x,route], axis=-1, name='concat_{}'.format(i))
                x = conv2d(x, self.filter[i], 1, name='block_{}_out'.format(i))

            if i==2 or i==3:
                out.append(x)

        ## spp
        x = conv2d(x, 512, 1, activation='leaky')
        x = conv2d(x, 1024, 3, activation='leaky')
        x = conv2d(x, 512, 1, activation='leaky')

        x = tf.concat([tf.nn.max_pool(x, ksize=13, padding='SAME', strides=1),
                   tf.nn.max_pool(x, ksize=9, padding='SAME', strides=1)
                      , tf.nn.max_pool(x, ksize=5, padding='SAME', strides=1), x], axis=-1)

        x = conv2d(x, 512, 1, activation='leaky')
        x = conv2d(x, 1024, 3, activation='leaky')
        x = conv2d(x, 512, 1, activation='leaky')
        out.append(x)
        if self.include_top:
            return tf.keras.Model(input, x)
        return tf.keras.Model(input,out)

    def head(self,x):
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(self.num_classes,activation='softmax')(x)
        return x


if __name__== '__main__':

    parser = argparse.ArgumentParser(description='Darknet53 implementation.')
    parser.add_argument('--batch_size', type=int, help = 'size of batch', default=4)
    parser.add_argument('--img_size',              type=int,   help='input height', default=512)
    args = parser.parse_args()
    darknet = CSPDarkNet53(args,1000)
    darknet.model.summary()


