from util import *
import argparse

def resblock(input,filter,i,j):
    x = conv2d(input,filter//2,1,name='block_{}_{}_conv1'.format(i,j),activation='leaky')
    x = conv2d(x,filter,3,name='block_{}_{}_conv3'.format(i,j),activation='leaky')
    return tf.keras.layers.Add()([input,x])

class DarkNet53(object):
    def __init__(self,args,num_classes):
        self.batch_size = args.batch_size
        self.input_shape = [args.img_h,args.img_w,3]
        self.filter = [64,128,256,512,1024]
        self.res_iter = [1,2,8,8,4]
        self.include_top = args.include_top
        self.num_classes = num_classes
        self.model = self.build()

    def build(self):
        model = self.backbone()
        if not self.include_top:
            return model
        return tf.keras.Model(model.input,self.head(model.output))

    def backbone(self):
        input = tf.keras.layers.Input(shape = (self.input_shape[0],self.input_shape[1],3))
        x = conv2d(input,32,3,activation='leaky')

        for i,iter in enumerate(self.res_iter):
            x  = conv2d(x , self.filter[i], 3, 2, name='block_{}_down'.format(i),activation='leaky')
            for j in range(iter):
                x = resblock(x, self.filter[i], i, j)

        return tf.keras.Model(input,x)

    def head(self,x):
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(self.num_classes,activation='softmax')(x)
        return x


if __name__== '__main__':

    parser = argparse.ArgumentParser(description='CSPDarknet53 implementation.')
    parser.add_argument('--batch_size', type=int, help = 'size of batch', default=4)
    parser.add_argument('--img_h',              type=int,   help='input height', default=256)
    parser.add_argument('--img_w',               type=int,   help='input width', default=256)
    parser.add_argument('--include_top', type=bool, help='include classifier? default is False',default=True)
    args = parser.parse_args()
    darknet = DarkNet53(args,1000)
    darknet.model.summary()


