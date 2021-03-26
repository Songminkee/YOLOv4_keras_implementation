# Disable tf log
import os
#os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from model import YOLOv4
from dataloader import loader
import numpy as np
import time
from util import load_darknet_weights,freeze_layer
import matplotlib.pyplot as plt

def train(args,hyp):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    train_set = loader.DataLoader(args, hyp, 'train')
    val_set = loader.DataLoader(args, hyp, 'val')
    val_len =int(np.ceil(len(val_set) / args.batch_size))
    accum_steps = max(round(args.update_batch / args.batch_size), 1)
    global_step = tf.Variable(1, trainable=False, dtype=tf.int64)
    best_loss = 9999999
    # print train info
    print('-' * 10)
    print('hyp\n',hyp)
    print('partial : {}'.format(args.partial))
    print('learning decay policy : {}'.format(args.policy))
    print('batch_size : {}'.format(args.batch_size))
    print('accum_steps : {}'.format(accum_steps))
    print('optimizer : {}'.format(args.optimizer))
    steps_per_epoch = int(np.ceil(len(train_set) / args.batch_size))
    if args.train_by_steps:
        print('warmup by steps : {}'.format(args.warmup_by_steps))
        if args.warmup_by_steps:
            warmup_steps = int(np.ceil(args.warmup_steps * args.update_batch /args.batch_size))
        else:
            warmup_steps = args.warmup_epochs * steps_per_epoch
            print('warmup epochs : {}'.format(args.warmup_epochs))
        if not warmup_steps:
            warmup_steps = 1
        print('warmup steps : {}'.format(warmup_steps))
        print('train by steps : {}'.format(args.train_by_steps))
        total_steps = int(np.ceil(args.train_steps * args.update_batch /args.batch_size))+warmup_steps
    else:
        print('warmup by steps : {}'.format(args.warmup_by_steps))
        if args.warmup_by_steps:
            warmup_steps = int(np.ceil(args.warmup_steps * args.update_batch /args.batch_size))
        else:
            warmup_steps = args.warmup_epochs * steps_per_epoch
            print('warmup epochs : {}'.format(args.warmup_epochs))
        if not warmup_steps:
            warmup_steps = 1
        print('warmup steps : {}'.format(warmup_steps))
        print('train by steps : {}'.format(args.train_by_steps))
        print('train epochs : {}'.format(args.epochs))
        total_steps = args.epochs * steps_per_epoch + warmup_steps
    print('total_steps : {}'.format(total_steps))
    
    out_type=[tf.float32]
    if args.is_tiny:
        YOLO = YOLOv4.YOLOv4_tiny(args,hyp)
        out_type= tuple(out_type*5)
    else:
        YOLO = YOLOv4.YOLOv4(args, hyp)
        out_type= tuple(out_type*7)
    
    train_set = tf.data.Dataset.from_generator(train_set,out_type)
    train_set = train_set.batch(args.batch_size).repeat(1)

    val_set = tf.data.Dataset.from_generator(val_set,out_type)
    val_set=val_set.batch(args.batch_size).repeat(1)

    # load pretrained model
    if args.weight_path!='':
        if args.is_darknet_weight:
            print('load darkent weight from {}'.format(args.weight_path))
            print('include_top = ',args.include_top)
            load_darknet_weights(YOLO.model, args.weight_path, args.is_tiny,args.include_top,args.partial)
        else:
            print('load_model from {}'.format(args.weight_path))
            YOLO.model.load_weights(args.weight_path)

    if not os.path.exists(args.weight_save_path):
        os.makedirs(args.weight_save_path)

    if args.optimizer=='sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate= hyp['lr0'] / warmup_steps,momentum= hyp['momentum'],nesterov=True)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=hyp['lr0']/warmup_steps)

    YOLO.model.compile(optimizer=optimizer)

    if args.summary:
        print('Summary will be stored in {} \n You can check the learning contents with the command "tensorboard --logdir {}"'.format(args.log_path,args.log_path))
        if not os.path.exists(args.log_path):
            os.makedirs(args.log_path)
        writer = tf.summary.create_file_writer(args.log_path)
    print('-' * 10)
    
    #train function
    @tf.function
    def train_step(images, labels,boxs, YOLO, accum_gradient,cnt):
        with tf.GradientTape() as tape:
            tape.watched_variables()
            iou_loss, object_loss, class_loss = 0,0,0
            pred, raw = YOLO.model(images, training=True)
            for i in range(len(labels)):
                l_iou, l_object, l_cls = YOLO.loss(pred[i],raw[i],labels[i],boxs[i],i,
                                                                    step=global_step if args.summary else None,
                                                                    writer=writer if args.summary else None)
                iou_loss += l_iou
                object_loss += l_object
                class_loss += l_cls
            loss_val =  iou_loss * hyp['giou'] + object_loss * hyp['obj'] + class_loss * hyp['cls']

            if args.summary:
                with writer.as_default():
                    tf.summary.scalar("iou_loss", iou_loss, step=global_step)
                    tf.summary.scalar("object_loss", object_loss, step=global_step)
                    tf.summary.scalar("class_loss", class_loss, step=global_step)
                    tf.summary.scalar("loss", loss_val, step=global_step)

            # accum gradient
            gradients = tape.gradient(loss_val, YOLO.model.trainable_variables)
            accum_gradient = [(acum_grad + grad) for acum_grad, grad in zip(accum_gradient, gradients)]


            # apply accum gradient
            if global_step % accum_steps == 0 or global_step == total_steps:
                accum_gradient = [this_grad/cnt for this_grad in accum_gradient] # /cnt
                cnt=0.
                YOLO.model.optimizer.apply_gradients(zip(accum_gradient, YOLO.model.trainable_variables))
                accum_gradient = [tf.zeros_like(this_var) for this_var in YOLO.model.trainable_variables]

                # learning rate schedule
                global_step.assign_add(1)
                if global_step <= warmup_steps:
                    lr = tf.cast( tf.pow(global_step / warmup_steps, 4) * hyp['lr0'],tf.float32) if args.dark_warmup else tf.cast(global_step / warmup_steps * hyp['lr0'],tf.float32)
                else:
                    if args.policy =='cosine':
                        lr = tf.cast(hyp['lrf'] + 0.5 * (hyp['lr0'] - hyp['lrf']) * (
                            (1 + tf.cos((global_step - warmup_steps) / (total_steps - warmup_steps) * np.pi))
                        ),tf.float32)
                    elif args.policy =='steps':
                        if global_step < int(total_steps*0.8): 
                            lr = tf.cast(hyp['lr0'],tf.float32)
                        elif global_step < int(total_steps*0.9):
                            lr = tf.cast(hyp['lr0'] * 0.1,tf.float32)
                        else:
                            lr = tf.cast(hyp['lr0'] * (0.1**2),tf.float32)
                YOLO.model.optimizer.lr.assign(lr)
            else:
                global_step.assign_add(1)
        
        return loss_val, accum_gradient, cnt

    # test_function
    @tf.function
    def test_step(images, labels,boxs,YOLO):
        iou_loss, object_loss, class_loss = 0,0,0
        pred, raw = YOLO.model(images, training=True)
        for i in range(len(labels)):
            l_iou, l_object, l_cls = YOLO.loss(pred[i],raw[i],labels[i],boxs[i],i,
                                                                step=global_step if args.summary else None,
                                                                writer=writer if args.summary else None)
            iou_loss += l_iou
            object_loss += l_object
            class_loss += l_cls
        loss_val =  iou_loss * hyp['giou'] + object_loss * hyp['obj'] + class_loss * hyp['cls']
        return loss_val


    # accum gradient init
    accum_gradient = [tf.zeros_like(this_var) for this_var in YOLO.model.trainable_variables]
    start_time = time.time()

    # total_epochs cal
    if args.train_by_steps:
        total_epochs = int(total_steps/steps_per_epoch+1)
    elif not args.train_by_steps and args.warmup_by_steps:
        total_epochs = int(warmup_steps/steps_per_epoch+1)+args.epochs
    else:
        total_epochs = args.warmup_epochs+args.epochs
    cnt=0.
    
    for epoch in range(total_epochs):
        if args.is_tiny:
            for images, s_label, l_label, s_box, l_box in train_set:
                labels = [s_label,l_label]
                boxs = [s_box,l_box]
                cnt+=1
                loss_val, accum_gradient, cnt = train_step(images, labels, boxs, YOLO, accum_gradient,cnt)
        else:
            for images, s_label, m_label, l_label, s_box, m_box, l_box in train_set:
                labels = [s_label,m_label,l_label]
                boxs = [s_box,m_box,l_box]
                cnt+=1
                loss_val, accum_gradient, cnt = train_step(images, labels, boxs, YOLO, accum_gradient,cnt)
            
        # log print
        time_sofar = (time.time() - start_time) / 3600
        training_time_left = (total_steps / global_step.numpy() - 1.0) * time_sofar
        print("=> Epoch:%4d | Train STEP %4d | loss_val: %4.2f | time elapsed: %4.2f h | time left: %4.2f h " % (epoch,global_step.numpy(), loss_val,time_sofar,training_time_left))

        # validation step
        validation_loss = 0
        val_cnt = 0

        if args.is_tiny:
            for val_images, s_label, l_label, s_box, l_box in train_set:
                val_labels = [s_label, l_label]
                val_boxs = [s_box, l_box]
                validation_loss += test_step(val_images, val_labels,val_boxs,YOLO)
                val_cnt+=1
        else:
            for val_images, s_label, m_label, l_label, s_box, m_box, l_box in train_set:
                val_labels = [s_label,m_label,l_label]
                val_boxs = [s_box,m_box,l_box]
                validation_loss += test_step(val_images, val_labels,val_boxs,YOLO)
                val_cnt+=1

        if args.summary:
            with writer.as_default():
                tf.summary.scalar("val_loss", validation_loss/val_cnt, step=global_step)

        # log print
        time_sofar = (time.time() - start_time) / 3600
        training_time_left = (total_steps / global_step.numpy() - 1.0) * time_sofar
        print("=> Epoch:%4d | VAL STEP %4d | loss_val: %4.2f | time elapsed: %4.2f h | time left: %4.2f h " % (
        epoch, global_step.numpy(), validation_loss/val_cnt, time_sofar, training_time_left))

        # save best model
        if best_loss> validation_loss:
            best_loss = validation_loss
            YOLO.model.save_weights(args.weight_save_path + '/best')

        # exit condition
        if global_step.numpy()==total_steps:
            break

        # save weight
        if epoch>0 and epoch%10==0:
            YOLO.model.save_weights(args.weight_save_path + '/{}'.format(epoch))

    # finish train
    YOLO.model.save_weights(args.weight_save_path+'/final')


if __name__== '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='YOLOv4 implementation.')
    parser.add_argument('--mode',
                        default='train',
                        const='train',
                        nargs='?',
                        choices=['train', 'test'],
                        help='Mode [train, test]')
    # batch
    parser.add_argument('--update_batch', type=int, help='Size of update batch size. ',
                        default=64)
    parser.add_argument('--batch_size', type=int, help = 'Size of batch size. (But, The gradient update is performed every update_batch/batch size.) / default : 4 / update_batch (YOLO v4/ YOLOv4 tiny)',
                        default=64)

    # train step or epochs
    parser.add_argument('--train_by_steps',action='store_true', help = 'Flag for whether to proceed with the training by step or epoch. / default : false')
    parser.add_argument('--train_steps', type=int, default=500500, help ='This is the total iteration to be carried out based on update_batch batches. So the total steps will be train_steps * update_batch / batch_size. Used only when the train_by_steps flag is true.  / default : 500500')
    parser.add_argument('--warmup_by_steps', action='store_true', help ='Flag for whether to proceed with the warm up by step or epoch. / default : false')
    parser.add_argument('--warmup_steps', type = int, default=1000, help = 'This is the total iteration of warm up to be carried out based on update_batch batches. So the steps will be warm up_steps * update_batch / batch_size. Used only when the warmup_by_steps flag is true. / default : 1000')
    
    parser.add_argument('--epochs', type=int,default=300 , help ='Total epochs to be trained. Used only when the train_by_steps flag is false. / default : 300')
    parser.add_argument('--warmup_epochs', type=int, default=3, help ='Total epochs to warm up. Used only when the warmup_by_steps flag is false. / default : 3')
    parser.add_argument('--save_steps', type=int, default=1000, help ='Step cycle to store weight. /default : 1000')
    
    # optimizer
    parser.add_argument('--optimizer',
                        default='sgd',
                        const='sgd',
                        nargs='?',
                        choices=['sgd', 'adam'],
                        help='optimizer [sgd, adam]')
    parser.add_argument('--policy',
                        default='steps',
                        const='steps',
                        nargs='?',
                        choices=['steps', 'cosine'],
                        help='learning decay policy [steps, cosine]')
    parser.add_argument('--dark_warmup', action='store_true',help='If true, the learning rate does not increase linearly during warm up, but increases by (now_step/warmup step)**4.')

    # model and weight 
    parser.add_argument('--is_tiny', action='store_true', help ='Flag for using tiny. / default : false')
    parser.add_argument('--is_darknet_weight', action='store_true',
                        help='If true, load the weight file used by the darknet framework.')

    parser.add_argument('--include_top', action='store_true',
                        help='If true, load all weight file used by the darknet framework. If the number of classes is different from coco, the last layers are not loaded. ')
    parser.add_argument('--partial',action='store_true',help='Train only last layer.')

    parser.add_argument('--weight_path',type=str,default='',help = 'Path of weight file / default : ""')
    
    parser.add_argument('--log_path', type=str, default='./log/', help = 'logdir path for Tensorboard')
    parser.add_argument('--summary', action='store_false')
    parser.add_argument('--weight_save_path', type=str, default='./weight',help = 'Path to store weights. / default : "./wegiht"')
      
    
    # train method
    parser.add_argument('--iou_method',
                        default='GIoU',
                        const='GIou',
                        nargs='?',
                        choices=['GIoU', 'CIoU','DIoU','IoU'])
    parser.add_argument('--focal', action='store_true')

    # data loader
    parser.add_argument('--img_size',              type=int,   help='Size of input image. / default : 416', default=416)
    parser.add_argument('--letter_box',action='store_false')
    parser.add_argument('--augment',              action='store_false',   help='Flag of augmentation (hsv, flip, random affine) / default : true')
    parser.add_argument('--mosaic', action='store_false', help='Flag of mosaic augmentation / default : true')
    parser.add_argument('--is_shuffle', action='store_false', help='Flag of data shuffle / default : true')
    parser.add_argument('--data_root',              type=str,   help='Root path of class name file and coco_%2017.txt / default : "./data"'
                                                                     , default='./data')
    parser.add_argument('--data_name', type=str,
                        help='Root path of class name file and coco_%2017.txt / default : "./data"'
                        , default='coco')
    parser.add_argument('--class_file',              type=str,   help='Class name file / default : "coco.name"', default='coco.names')
    parser.add_argument('--num_classes', type=int, help='Number of classes (in COCO 80) / default : 80', default=80)
    
    args = parser.parse_args()
    hyp = {'giou': 1.00,#3.54,  # giou loss gain
           'cls': 1.0,#37.4,  # cls loss gain
           'cls_pw' : 1.0,
           'obj': 1.0,#83.59,  # obj loss gain (=64.3*img_size/320 if img_size != 320)
           'obj_pw' : 1.0,
           'iou_t': 1.0,  # iou training threshold
           'lr0': 0.0013,  # initial learning rate (SGD=1.3E-3, Adam=1.3E-4)
           'lrf': 0.00013,  # final learning rate (with cos scheduler)
           'momentum': 0.949,  # SGD momentum
           'fl_gamma': 2.0,  # focal loss gamma (efficientDet default is gamma=1.5)
           'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
           'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
           'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
           'degrees': 0,#1.98,#10.0,#1.98 * 0,  # image rotation (+/- deg)
           'translate':0.1, #0.1,#0.05 * 0,  # image translation (+/- fraction)
           'scale':0.5,  # image scale (+/- gain)
           'shear':0,# 0.1,#0.641 * 0}  # image shear (+/- deg)
           'ignore_threshold': 0.7,
           'border' : 2,
           'flip_lr' : 0.5,
           'flip_ud' : 0.0,
           'soft' : 0.0
           }


    train(args,hyp)