from util import *
from model import CSPDarkNet53
from model import CSPDarkNet53_tiny

class YOLOv4(object):
    def __init__(self,args,hyp,stride=None,anchor=None,sigmoid_scale=None):
        if stride:
            self.stride = stride
        else:
            self.stride=default_stride()

        if anchor:
            self.anchor = anchor
        else:
            self.anchor = default_anchor()

        if sigmoid_scale:
            self.sigmoid_scale = sigmoid_scale
        else:
            self.sigmoid_scale = default_sigmoid_scale()

        self.gr = 0.02
        self.hyp = hyp
        self.batch_size = args.batch_size
        self.soft = args.soft
        self.anchors = make_anchor(self.stride,self.anchor)
        self.num_classes = args.num_classes
        self.backbone = CSPDarkNet53.CSPDarkNet53(args).model
        self.box_feature = self.head(self.backbone.output)
        self.out = self.pred(self.box_feature)
        self.model = tf.keras.Model(inputs=self.backbone.input,outputs=self.out)

    def head(self, backbone_out):
        r3, r2, r1 = backbone_out

        x = conv2d(r1, 256, 1, activation='leaky')
        x = upsample(x)
        x = tf.concat([conv2d(r2, 256, 1, activation='leaky'),x], -1)
        route1 = convset(x, 256)

        x = conv2d(route1, 128, 1, activation='leaky')
        x = upsample(x)
        x = tf.concat([conv2d(r3, 128, 1, activation='leaky'),x], -1)
        route2 = convset(x, 128)
        box1 = conv2d(route2, 256, 3, activation='leaky')
        box1 = tf.keras.layers.Conv2D(3 * (self.num_classes + 5), 1,
                                      kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                      kernel_initializer=tf.random_normal_initializer(stddev=0.01))(box1)

        x = tf.concat([conv2d(route2, 256, 3, 2, activation='leaky'),route1], -1)
        route3 = convset(x, 256)
        box2 = conv2d(route3, 512, 3, activation='leaky')
        box2 = tf.keras.layers.Conv2D(3 * (self.num_classes + 5), 1,
                                      kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                      kernel_initializer=tf.random_normal_initializer(stddev=0.01)
                                      )(box2)

        x = tf.concat([ conv2d(route3, 512, 3, 2, activation='leaky'),r1], -1)
        x = convset(x, 512)
        box3 = conv2d(x, 1024, 3, activation='leaky')
        box3 = tf.keras.layers.Conv2D(3 * (self.num_classes + 5), 1,
                                      kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                      kernel_initializer=tf.random_normal_initializer(stddev=0.01)
                                      )(box3)

        return [box1, box2, box3]

    def pred(self, boxes):
        pred = []
        for i, box in enumerate(boxes):
            box_shape = box.shape
            box = tf.reshape(box, (-1, box_shape[1], box_shape[2], 3, self.num_classes + 5))

            xy, wh, conf, cls = tf.split(box, ([2, 2, 1, self.num_classes]), -1)
            shape = tf.shape(xy)

            xy_grid = tf.meshgrid(tf.range(shape[2]), tf.range(shape[1]))  # w,h
            xy_grid = tf.expand_dims(tf.stack(xy_grid, -1), 2)
            xy_grid = tf.cast(tf.tile(tf.expand_dims(xy_grid, axis=0), [shape[0], 1, 1, 3, 1]), tf.float32)  # b,h,w,3,2

            pred_xy = ((tf.sigmoid(xy) * self.sigmoid_scale[i])  + xy_grid)
            pred_wh = tf.exp(wh) * self.anchors[i]
            pred_cls = tf.sigmoid(cls)
            pred_conf = tf.sigmoid(conf)
            pred.append(tf.concat([pred_xy, pred_wh, pred_conf, pred_cls], -1))

        return pred

    def loss(self,box_label,out,step=None,writer=None):
        bce = tf.keras.losses.BinaryCrossentropy(False, reduction=tf.keras.losses.Reduction.NONE)
        cp,cn = smoothing_value(self.num_classes,self.soft)
        iou_loss, object_loss, class_loss = 0, 0, 0
        for i in range(3):
            scaler = tf.stop_gradient(label_scaler(out[i]))
            label = box_label*scaler
            mask,idx,label = build_target(self.anchors[i],label,self.hyp)

            c_label, box = tf.split(label,[1,4],-1)

            pred = tf.gather_nd(out[i],idx)
            xywh,conf,cls = tf.split(pred,[4,1,self.num_classes],-1)

            # get giou or ciou or diou
            iou = tf.clip_by_value(get_iou_loss(xywh,box),0.0,1.0) # [b,max_label,3,1]

            # get obj(confidence) loss by iou
            l_obj = tf.expand_dims(bce((1.0-self.gr) + self.gr*iou, conf),-1) # [b,max_label,3,1]

            # get class_loss
            c_label = tf.one_hot(tf.cast(tf.squeeze(c_label),tf.int32), self.num_classes, on_value=cp, off_value=cn)
            l_cls = tf.expand_dims(bce(c_label,cls),-1) # [b,max_label,3,1]

            mask_num = tf.reduce_sum(tf.cast(mask,tf.float32))
            l_iou = tf.reduce_sum(tf.where(mask,1-iou,0))/(mask_num+1e-16)
            l_obj = tf.reduce_sum(tf.where(mask,l_obj,0))/(mask_num+1e-16)
            l_cls = tf.reduce_sum(tf.where(mask,l_cls,0))/(mask_num+1e-16)

            iou_loss += l_iou
            object_loss += l_obj
            class_loss += l_cls

            if writer != None:
                with writer.as_default():
                    tf.summary.scalar("l_iou_{}".format(i), l_iou, step=step)
                    tf.summary.scalar("l_obj_{}".format(i), l_obj, step=step)
                    tf.summary.scalar("l_cls_{}".format(i), l_cls, step=step)

        loss = iou_loss*self.hyp['giou'] + object_loss * self.hyp['obj'] + class_loss * self.hyp['cls']
        if writer != None:
            with writer.as_default():
                tf.summary.scalar("iou_loss", iou_loss, step=step)
                tf.summary.scalar("object_loss", object_loss, step=step)
                tf.summary.scalar("class_loss", class_loss, step=step)
                tf.summary.scalar("loss",class_loss,step=step)

        return loss *  self.batch_size / 64


class YOLOv4_tiny(object):
    def __init__(self,args,hyp,stride=None,anchor=None,sigmoid_scale=None):
        if stride:
            self.stride = stride
        else:
            self.stride=default_stride(is_tiny=args.is_tiny)

        if anchor:
            self.anchor = anchor
        else:
            self.anchor = default_anchor(is_tiny=args.is_tiny)

        if sigmoid_scale:
            self.sigmoid_scale = sigmoid_scale
        else:
            self.sigmoid_scale = default_sigmoid_scale(is_tiny=args.is_tiny)

        self.gr = 0.02
        self.hyp = hyp
        self.batch_size = args.batch_size
        self.soft = args.soft
        self.anchors = make_anchor(self.stride,self.anchor,is_tiny=args.is_tiny)
        self.num_classes = args.num_classes
        self.backbone = CSPDarkNet53_tiny.CSPDarkNet53_tiny(args).model
        self.box_feature = self.head(self.backbone.output)
        self.out = self.pred(self.box_feature)
        self.model = tf.keras.Model(inputs=self.backbone.input,outputs=self.out)

    def head(self, backbone_out):
        r1, r2 = backbone_out

        x = conv2d(r2, 256, 1, activation='leaky')

        box2 = conv2d(x, 512, 3, activation='leaky')
        box2 = tf.keras.layers.Conv2D(3 * (self.num_classes + 5), 1,
                                      kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                      kernel_initializer=tf.random_normal_initializer(stddev=0.01)
                                      )(box2)

        x = conv2d(x, 128, 1, activation='leaky')
        x = upsample(x)
        x = tf.concat([x, r1], -1)
        box1 = conv2d(x, 256, 3, activation='leaky')
        box1 = tf.keras.layers.Conv2D(3 * (self.num_classes + 5), 1,
                                      kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                      kernel_initializer=tf.random_normal_initializer(stddev=0.01)
                                      )(box1)

        return [box1, box2]

    def pred(self,boxes):
        pred = []

        for i,box in enumerate(boxes):
            box_shape = box.shape
            box = tf.reshape(box,(-1,box_shape[1],box_shape[2],3,self.num_classes+5))

            xy,wh,conf,cls = tf.split(box,([2,2,1,self.num_classes]),-1)
            shape = tf.shape(xy)

            xy_grid = tf.meshgrid(tf.range(shape[2]), tf.range(shape[1])) # w,h
            xy_grid = tf.expand_dims(tf.stack(xy_grid,-1),2)
            xy_grid = tf.cast(tf.tile(tf.expand_dims(xy_grid, axis=0), [shape[0], 1, 1, 3, 1]),tf.float32) # b,h,w,3,2

            pred_xy = ((tf.sigmoid(xy)*self.sigmoid_scale[i])+xy_grid)
            pred_wh = tf.exp(wh)*self.anchors[i]

            pred_cls = tf.sigmoid(cls)
            pred_conf = tf.sigmoid(conf)
            pred.append(tf.concat([pred_xy,pred_wh,pred_conf,pred_cls],-1))

        return pred

    def loss(self,box_label,out,step=None,writer=None):
        bce = tf.keras.losses.BinaryCrossentropy(False, reduction=tf.keras.losses.Reduction.NONE)
        cp,cn = smoothing_value(self.num_classes,self.soft)
        iou_loss, object_loss, class_loss = 0, 0, 0
        for i in range(2):
            scaler = tf.stop_gradient(label_scaler(out[i]))
            label = box_label*scaler
            mask,idx,label = build_target(self.anchors[i],label,self.hyp)

            c_label, box = tf.split(label,[1,4],-1)

            pred = tf.gather_nd(out[i],idx)
            xywh,conf,cls = tf.split(pred,[4,1,self.num_classes],-1)

            # get giou or ciou or diou
            iou = tf.clip_by_value(get_iou_loss(xywh,box),0.0,1.0) # [b,max_label,3,1]

            # get obj(confidence) loss by iou
            l_obj = tf.expand_dims(bce((1.0-self.gr) + self.gr*iou, conf),-1) # [b,max_label,3,1]

            # get class_loss
            c_label = tf.one_hot(tf.cast(tf.squeeze(c_label),tf.int32), self.num_classes, on_value=cp, off_value=cn)
            l_cls = tf.expand_dims(bce(c_label,cls),-1) # [b,max_label,3,1]

            mask_num = tf.reduce_sum(tf.cast(mask,tf.float32))
            l_iou = tf.reduce_sum(tf.where(mask,1-iou,0))/(mask_num+1e-16)
            l_obj = tf.reduce_sum(tf.where(mask,l_obj,0))/(mask_num+1e-16)
            l_cls = tf.reduce_sum(tf.where(mask,l_cls,0))/(mask_num+1e-16)

            iou_loss += l_iou
            object_loss += l_obj
            class_loss += l_cls

            if writer != None:
                with writer.as_default():
                    tf.summary.scalar("l_iou_{}".format(i), l_iou, step=step)
                    tf.summary.scalar("l_obj_{}".format(i), l_obj, step=step)
                    tf.summary.scalar("l_cls_{}".format(i), l_cls, step=step)

        loss = iou_loss*self.hyp['giou'] + object_loss * self.hyp['obj'] + class_loss * self.hyp['cls']
        if writer != None:
            with writer.as_default():
                tf.summary.scalar("iou_loss", iou_loss, step=step)
                tf.summary.scalar("object_loss", object_loss, step=step)
                tf.summary.scalar("class_loss", class_loss, step=step)
                tf.summary.scalar("loss",class_loss,step=step)

        return loss *  self.batch_size / 64



if __name__== '__main__':
    import argparse

    hyp = {'giou': 3.54,  # giou loss gain
           'cls': 37.4,  # cls loss gain
           'obj': 102.88,  # obj loss gain (=64.3*img_size/320 if img_size != 320)
           'iou_t': 0.213,  # iou training threshold
           'lr0': 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)
           'lrf': 0.0005,  # final learning rate (with cos scheduler)
           'momentum': 0.949,  # SGD momentum
           'weight_decay': 0.000484,  # optimizer weight decay
           'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
           'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
           'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
           'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
           'degrees': 1.98 * 0,  # image rotation (+/- deg)
           'translate': 0.05 * 0,  # image translation (+/- fraction)
           'scale': 0.5,  # image scale (+/- gain)
           'shear': 0.641 * 0}  # image shear (+/- deg)


    parser = argparse.ArgumentParser(description='YOLOv4 implementation.')
    parser.add_argument('--batch_size', type=int, help = 'size of batch', default=4)
    parser.add_argument('--img_size',              type=int,   help='input height', default=512)
    parser.add_argument('--num_classes', type=int, help='number of class', default=80)
    args = parser.parse_args()
    YOLO = YOLOv4(args,hyp)
    YOLO.model.summary()
