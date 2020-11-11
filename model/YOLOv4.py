from util import *
from model import CSPDarkNet53

class YOLOv4(object):
    def __init__(self,args):
        self.num_classes = args.num_classes
        self.backbone = CSPDarkNet53.CSPDarkNet53(args).model
        self.model = tf.keras.Model(self.backbone.input,self.head(self.backbone.output))

    def head(self,backbone_out):
        r3,r2,r1 = backbone_out

        x = conv2d(r1,256,1,activation='leaky')
        x = upsample(x)
        x = tf.concat([x,conv2d(r2,256,1,activation='leaky')],-1)
        route1 = convset(x,256)

        x = conv2d(route1,128,1,activation='leaky')
        x = upsample(x)
        x = tf.concat([x,conv2d(r3,128,1,activation='leaky')],-1)
        route2 = convset(x,128)

        x = tf.concat([route1,conv2d(route2,256,3,2,activation='leaky')],-1)
        route3 = convset(x,256)

        x = tf.concat([r1,conv2d(route3,512,3,2,activation='leaky')],-1)
        x = convset(x,512)

        box1 = conv2d(route2,256,3,activation='leaky')
        box1 = tf.keras.layers.Conv2D(3 * (self.num_classes + 5), 1, use_bias=False)(box1)
        box2 = conv2d(route3,512,3,activation='leaky')
        box2 = tf.keras.layers.Conv2D(3 * (self.num_classes + 5), 1, use_bias=False)(box2)
        box3 = conv2d(x,1024,3,activation='leaky')
        box3 = tf.keras.layers.Conv2D(3 * (self.num_classes + 5), 1, use_bias=False)(box3)

        return [box1,box2,box3]


if __name__== '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='YOLOv4 implementation.')
    parser.add_argument('--batch_size', type=int, help = 'size of batch', default=4)
    parser.add_argument('--img_h',              type=int,   help='input height', default=512)
    parser.add_argument('--img_w',               type=int,   help='input width', default=512)
    parser.add_argument('--num_classes', type=int, help='number of class', default=1000)
    args = parser.parse_args()
    YOLO = YOLOv4(args)
    YOLO.model.summary()
