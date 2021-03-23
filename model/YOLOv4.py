import sys
sys.path.append('/mnt/1E5E8DB55E8D866D/c_yolo')
from util import *
from model import CSPDarkNet53
from model import CSPDarkNet53_tiny

class YOLOv4(object):
    def __init__(self,args,hyp=None,stride=None,anchor=None,sigmoid_scale=None):
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

        self.img_size = args.img_size
        self.mode = args.mode
        self.hyp = hyp
        self.args = args
        self.anchors = make_anchor(self.stride,self.anchor)
        self.num_classes = args.num_classes
        self.backbone = CSPDarkNet53.CSPDarkNet53(args).model
        self.box_feature = self.head(self.backbone.output)
        self.head_model = tf.keras.Model(inputs=self.backbone.input,outputs=self.box_feature)
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
        box1 = tf.keras.layers.Conv2D(3 * (self.num_classes + 5) ,1,#if self.num_classes>1 else 3*5, 1,
                                      kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                      #kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                        bias_initializer=tf.constant_initializer(0.),
                                      kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.002, maxval=0.002)#tf.random_normal_initializer(stddev=0.01)
                                      )(box1)
        
        x = tf.concat([conv2d(route2, 256, 3, 2, activation='leaky'),route1], -1)
        route3 = convset(x, 256)
        box2 = conv2d(route3, 512, 3, activation='leaky')
        box2 = tf.keras.layers.Conv2D(3 * (self.num_classes + 5) ,1,#if self.num_classes>1 else 3*5, 1,
                                      kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                      #kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                        bias_initializer=tf.constant_initializer(0.),
                                       kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.002, maxval=0.002)#tf.random_normal_initializer(stddev=0.01)
                                      )(box2)

        x = tf.concat([ conv2d(route3, 512, 3, 2, activation='leaky'),r1], -1)
        x = convset(x, 512)
        box3 = conv2d(x, 1024, 3, activation='leaky')
        box3 = tf.keras.layers.Conv2D(3 * (self.num_classes + 5) ,1,#if self.num_classes>1 else 3*5, 1,
                                        kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                        #kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                        bias_initializer=tf.constant_initializer(0.),
                                      kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.002, maxval=0.002)#tf.random_normal_initializer(stddev=0.01)
                                      )(box3)
        return [box1,box2,box3]

    def pred(self, boxes):
        pred = []
        if self.mode!='eval':
            raw = []
        for i, box in enumerate(boxes):
            shape = tf.shape(box)
            grid = self.img_size//self.stride[i]
            box = tf.reshape(box, (shape[0], grid, grid, 3, self.num_classes + 5 ))#if self.num_classes>1 else 5))

            xy, wh, conf, cls = tf.split(box, ([2, 2, 1, self.num_classes]), -1)
            if self.mode!='eval':
                raw.append(tf.concat([conf, cls],axis=-1))

            pred_cls = tf.sigmoid(cls)
            
            xy_grid = tf.meshgrid(tf.range(grid), tf.range(grid))  # w,h
            xy_grid = tf.expand_dims(tf.stack(xy_grid, -1), 2)
            xy_grid = tf.cast(tf.tile(tf.expand_dims(xy_grid, axis=0), [shape[0], 1, 1, 3, 1]), tf.float32)  # b,h,w,3,2

            pred_xy = ((tf.sigmoid(xy)*self.sigmoid_scale[i])-0.5*(self.sigmoid_scale[i]-1)+xy_grid)
            pred_wh = (tf.exp(wh) * self.anchors[i])
            if self.mode!='eval':
                pred_xy  *= self.stride[i]
                pred_wh *= self.stride[i
                ] 
            pred_conf = tf.sigmoid(conf)
            pred.append(tf.concat([pred_xy, pred_wh, pred_conf, pred_cls], -1))
        if self.mode!='eval':
            return pred, raw
        return pred


    def loss(self,pred,raw,label,box,i,step=None,writer=None):
        shape = tf.shape(pred)
        BATCH = tf.cast(shape,tf.float32)[0]
        raw_conf, raw_cls = tf.split(raw,[1,self.num_classes],axis=-1)
        pred_box,pred_conf,__ = tf.split(pred,[4,1,self.num_classes],axis=-1)
        label_box,obj_mask,label_cls = tf.split(label,[4,1,self.num_classes],axis=-1)

        giou = tf.expand_dims(get_iou_loss(pred_box, label_box,method='CIoU'),axis=-1)
        #bbox_loss_scale = 2.0 - 1.0 * label_box[:, :, :, :, 2:3] * label_box[:, :, :, :, 3:4] / (self.img_size ** 2)
        l_iou = obj_mask  * (1-giou) #* bbox_loss_scale

        iou = get_iou_loss(pred_box[:,:,:,:,np.newaxis,:],box[:,np.newaxis,np.newaxis,np.newaxis,:,:],method='IoU')
        max_iou = tf.expand_dims(tf.reduce_max(iou,axis=-1),axis=-1)
        no_obj_mask = (1.0 - obj_mask) * tf.cast( max_iou < self.hyp['ignore_threshold'], tf.float32 )

        l_obj = obj_mask * tf.nn.weighted_cross_entropy_with_logits(labels=obj_mask,logits=raw_conf,pos_weight=self.hyp['obj_pw']) #tf.nn.sigmoid_cross_entropy_with_logits(labels=obj_mask, logits=raw_conf)
        l_noobj = no_obj_mask *  tf.nn.weighted_cross_entropy_with_logits(labels=obj_mask,logits=raw_conf,pos_weight=self.hyp['obj_pw']) #tf.nn.sigmoid_cross_entropy_with_logits(labels=obj_mask, logits=raw_conf)
        
        l_object = l_obj + l_noobj
        
        if self.args.focal:
            conf_focal = tf.pow(obj_mask - pred_conf, self.hyp['fl_gamma'])
            l_object = l_object * conf_focal

        l_cls = obj_mask * tf.nn.weighted_cross_entropy_with_logits(labels=label_cls,logits=raw_cls,pos_weight=self.hyp['cls_pw'])#tf.nn.sigmoid_cross_entropy_with_logits(labels=label_cls, logits=raw_cls)

        if writer != None:
            with writer.as_default():
                tf.summary.scalar("l_iou_{}".format(i), tf.reduce_sum(l_iou), step=step)
                tf.summary.scalar("l_obj_{}".format(i), tf.reduce_sum(l_obj), step=step)
                tf.summary.scalar("l_noobj_{}".format(i), tf.reduce_sum(l_noobj), step=step)
                tf.summary.scalar("l_cls_{}".format(i), tf.reduce_sum(l_cls), step=step)
                tf.summary.scalar("num_noobj_{}".format(i),tf.reduce_sum(no_obj_mask), step=step)
                tf.summary.scalar("num_thre_{}".format(i),tf.reduce_sum(1-(no_obj_mask+obj_mask)), step=step)
                tf.summary.scalar("num_obj_{}".format(i),tf.reduce_sum(obj_mask), step=step)
        
        l_iou = tf.reduce_mean(tf.reduce_sum(l_iou,axis=[1,2,3,4]))
        l_object = tf.reduce_mean(tf.reduce_sum(l_object,axis=[1,2,3,4]))
        l_cls = tf.reduce_mean(tf.reduce_sum(l_cls,axis=[1,2,3,4]))
        
        return l_iou, l_object, l_cls


class YOLOv4_tiny(object):
    def __init__(self,args,hyp=None,stride=None,anchor=None,sigmoid_scale=None):
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

        self.img_size = args.img_size
        self.mode = args.mode
        self.hyp = hyp
        self.args = args
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
        if self.mode!='eval':
            raw = []
        for i,box in enumerate(boxes):
            shape = tf.shape(box)
            grid = self.img_size//self.stride[i]
            box = tf.reshape(box, (shape[0], grid, grid, 3, self.num_classes + 5 ))#if self.num_classes>1 else 5))

            xy,wh,conf,cls = tf.split(box,([2,2,1,self.num_classes]),-1)
            if self.mode!='eval':
                raw.append(tf.concat([conf, cls],axis=-1))

            pred_cls = tf.sigmoid(cls)

            xy_grid = tf.meshgrid(tf.range(shape[2]), tf.range(shape[1])) # w,h
            xy_grid = tf.expand_dims(tf.stack(xy_grid,-1),2)
            xy_grid = tf.cast(tf.tile(tf.expand_dims(xy_grid, axis=0), [shape[0], 1, 1, 3, 1]),tf.float32) # b,h,w,3,2

            pred_xy = ((tf.sigmoid(xy)*self.sigmoid_scale[i])-0.5*(self.sigmoid_scale[i]-1)+xy_grid)
            pred_wh = tf.exp(wh)*self.anchors[i]
            if self.mode!='eval':
                pred_xy  *= self.stride[i]
                pred_wh *= self.stride[i] 

            pred_conf = tf.sigmoid(conf)
            pred.append(tf.concat([pred_xy,pred_wh,pred_conf,pred_cls],-1))
        if self.mode!='eval':
            return pred, raw
        return pred

    def loss(self,pred,raw,label,box,i,step=None,writer=None):
        shape = tf.shape(pred)
        BATCH = tf.cast(shape,tf.float32)[0]
        raw_conf, raw_cls = tf.split(raw,[1,self.num_classes],axis=-1)
        pred_box,pred_conf,__ = tf.split(pred,[4,1,self.num_classes],axis=-1)
        label_box,obj_mask,label_cls = tf.split(label,[4,1,self.num_classes],axis=-1)

        giou = tf.expand_dims(get_iou_loss(pred_box, label_box,method='CIoU'),axis=-1)
        #bbox_loss_scale = 2.0 - 1.0 * label_box[:, :, :, :, 2:3] * label_box[:, :, :, :, 3:4] / (self.img_size ** 2)
        l_iou = obj_mask  * (1-giou) #* bbox_loss_scale

        iou = get_iou_loss(pred_box[:,:,:,:,np.newaxis,:],box[:,np.newaxis,np.newaxis,np.newaxis,:,:],method='IoU')
        max_iou = tf.expand_dims(tf.reduce_max(iou,axis=-1),axis=-1)
        no_obj_mask = (1.0 - obj_mask) * tf.cast( max_iou < self.hyp['ignore_threshold'], tf.float32 )

        l_obj = obj_mask * tf.nn.weighted_cross_entropy_with_logits(labels=obj_mask,logits=raw_conf,pos_weight=self.hyp['obj_pw']) #tf.nn.sigmoid_cross_entropy_with_logits(labels=obj_mask, logits=raw_conf)
        l_noobj = no_obj_mask *  tf.nn.weighted_cross_entropy_with_logits(labels=obj_mask,logits=raw_conf,pos_weight=self.hyp['obj_pw']) #tf.nn.sigmoid_cross_entropy_with_logits(labels=obj_mask, logits=raw_conf)
        
        l_object = l_obj + l_noobj
        
        if self.args.focal:
            conf_focal = tf.pow(obj_mask - pred_conf, self.hyp['fl_gamma'])
            l_object = l_object * conf_focal

        l_cls = obj_mask * tf.nn.weighted_cross_entropy_with_logits(labels=label_cls,logits=raw_cls,pos_weight=self.hyp['cls_pw'])#tf.nn.sigmoid_cross_entropy_with_logits(labels=label_cls, logits=raw_cls)

        if writer != None:
            with writer.as_default():
                tf.summary.scalar("l_iou_{}".format(i), tf.reduce_sum(l_iou), step=step)
                tf.summary.scalar("l_obj_{}".format(i), tf.reduce_sum(l_obj), step=step)
                tf.summary.scalar("l_noobj_{}".format(i), tf.reduce_sum(l_noobj), step=step)
                tf.summary.scalar("l_cls_{}".format(i), tf.reduce_sum(l_cls), step=step)
                tf.summary.scalar("num_noobj_{}".format(i),tf.reduce_sum(no_obj_mask), step=step)
                tf.summary.scalar("num_thre_{}".format(i),tf.reduce_sum(1-(no_obj_mask+obj_mask)), step=step)
                tf.summary.scalar("num_obj_{}".format(i),tf.reduce_sum(obj_mask), step=step)
        
        l_iou = tf.reduce_mean(tf.reduce_sum(l_iou,axis=[1,2,3,4]))
        l_object = tf.reduce_mean(tf.reduce_sum(l_object,axis=[1,2,3,4]))
        l_cls = tf.reduce_mean(tf.reduce_sum(l_cls,axis=[1,2,3,4]))
        
        return l_iou, l_object, l_cls



if __name__== '__main__':
    import argparse

    hyp = {'giou': 1.0,#3.54,  # giou loss gain
           'cls': 1.0,#37.4,  # cls loss gain
           'cls_pw' : 1.0,
           'obj': 1.0,#83.59,  # obj loss gain (=64.3*img_size/320 if img_size != 320)
           'obj_pw' : 1.0,
           'iou_t': 0.213,  # iou training threshold
           'lr0': 0.0013,  # initial learning rate (SGD=5E-3, Adam=5E-4)
           'lrf': 0.00013,  # final learning rate (with cos scheduler)
           'momentum': 0.949,  # SGD momentum
           'fl_gamma': 2.0,  # focal loss gamma (efficientDet default is gamma=1.5)
           'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
           'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
           'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
           'degrees': 0,#1.98,#10.0,#1.98 * 0,  # image rotation (+/- deg)
           'translate':0.5, #0.1,#0.05 * 0,  # image translation (+/- fraction)
           'scale':0.1,  # image scale (+/- gain)
           'shear':0,# 0.1,#0.641 * 0}  # image shear (+/- deg)
           'ignore_threshold': 0.7,
           'border' : 2,
           'flip_lr' : 0.5,
           'flip_ud' : 0.0,
           'soft' : 0.0
           }


    parser = argparse.ArgumentParser(description='YOLOv4 implementation.')
    parser.add_argument('--batch_size', type=int, help = 'size of batch', default=4)
    parser.add_argument('--img_size',              type=int,   help='input height', default=416)
    parser.add_argument('--num_classes', type=int, help='number of class', default=1)
    parser.add_argument('--is_tiny', action='store_false')
    parser.add_argument('--soft', type=float, help='number of class', default=0.0)
    parser.add_argument('--partial', action='store_false')
    parser.mode='test'
    args = parser.parse_args()
    args.mode='test'
    YOLO = YOLOv4_tiny(args,hyp)
    YOLO.model.summary()
    #print(YOLO.head)
    load_darknet_weights(YOLO.model,'./dark_weight/yolov4-tiny.weights',True,True)
    
    for layer in YOLO.model.layers:
        print(layer.name,layer.trainable)
        if ('batch_normalization' not in layer.name and 'conv2d' not in layer.name):
            print(layer.trainable_variables)
    #YOLO.model.summary()
    input_data = np.array(np.random.random_sample([4,416,416,3]), dtype=np.float32)
    #a,b = YOLO.head_model(input_data)
    #print(a)
    #print(b)