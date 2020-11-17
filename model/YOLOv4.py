from util import *
from model import CSPDarkNet53

class YOLOv4(object):
    def __init__(self,args,stride=None,anchor=None,sigmoid_scale=None):
        if stride:
            self.stride = stride
        else:
            self.stride=self.default_stride()

        if anchor:
            self.anchor = anchor
        else:
            self.anchor = self.default_anchor()

        if sigmoid_scale:
            self.sigmoid_scale = sigmoid_scale
        else:
            self.sigmoid_scale = self.default_sigmoid_scale()

        self.anchors = make_anchor(self.stride,self.anchor)
        self.num_classes = args.num_classes
        self.backbone = CSPDarkNet53.CSPDarkNet53(args).model
        self.box_feature = self.head(self.backbone.output)
        self.out = self.pred(self.box_feature)
        self.model = tf.keras.Model(self.backbone.input,self.out)

    def default_stride(self):
        return np.array([8,16,32])

    def default_anchor(self):
        return np.array([12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401])

    def default_sigmoid_scale(self):
        return np.array([1.2, 1.1, 1.05])

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

    def pred(self,boxes):
        pred = []
        for i,box in enumerate(boxes):
            box_shape = tf.shape(box)

            box = tf.reshape(box,(box_shape[0],box_shape[1],box_shape[2],3,self.num_classes+5))

            conf,xy,wh,cls = tf.split(box,([1,2,2,self.num_classes]),-1)

            shape = tf.shape(xy)

            xy_grid = tf.meshgrid(tf.range(shape[2]), tf.range(shape[1])) # w,h
            xy_grid = tf.expand_dims(tf.stack(xy_grid,-1),2)
            xy_grid = tf.cast(tf.tile(tf.expand_dims(xy_grid, axis=0), [shape[0], 1, 1, 3, 1]),tf.float32) # b,h,w,3,2

            pred_xy = (tf.sigmoid(xy)*self.sigmoid_scale[i]-0.5*(self.sigmoid_scale[i]-1)+xy_grid)*self.stride[i]
            pred_wh = tf.exp(wh)*self.anchor[i]
            pred_xywh = tf.concat([pred_xy,pred_wh],-1)

            pred_cls = tf.sigmoid(cls)
            pred_conf = tf.sigmoid(conf)
            pred.append(tf.concat([pred_xywh,pred_conf,pred_cls],-1))

        return pred

if __name__== '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='YOLOv4 implementation.')
    parser.add_argument('--batch_size', type=int, help = 'size of batch', default=4)
    parser.add_argument('--img_size',              type=int,   help='input height', default=512)
    parser.add_argument('--num_classes', type=int, help='number of class', default=1000)
    args = parser.parse_args()
    YOLO = YOLOv4(args)
    YOLO.model.summary()
