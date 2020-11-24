import tensorflow as tf
from util import *
import argparse

def tiny_cspblock(input,filter):
    route = input
    x =  tf.split(input, num_or_size_splits=2, axis=-1)[1]
    x = conv2d(x,filter,3,activation='leaky')
    route_1 = x
    x = conv2d(x,filter,3,activation='leaky')
    x = tf.concat([x,route_1],-1)
    x = conv2d(x,filter*2,1,activation='leaky')
    return route,x

class CSPDarkNet53_tiny(object):
    def __init__(self,args,num_classes=None,include_top=False):
        self.batch_size = args.batch_size
        self.input_shape = [args.img_size,args.img_size,3]
        self.filter = [32,64,128]
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
        x = conv2d(input, 32, 3,2,activation='leaky')
        x = conv2d(x,64,3,2,activation='leaky')
        x = conv2d(x,64,3,activation='leaky')
        out = []

        for i, filter in enumerate(self.filter):
            r,x = tiny_cspblock(x,filter)
            if i==2:
                out.append(x)
            x = tf.concat([r, x], axis=-1)
            x = tf.keras.layers.MaxPool2D(2, 2, 'same')(x)
            x = conv2d(x,filter*4,3,activation='leaky')

        out.append(x)

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


