# Disable tf log
import os
#os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow_addons as tfa
from model import YOLOv4
from dataloader import loader
import numpy as np
import time
import gc

#print(tf.config.experimental.list_physical_devices('GPU'))

def train(args,hyp):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    train_set = loader.DataLoader(args, hyp, 'train')
    val_set = loader.DataLoader(args, hyp, 'val')
    val_len =int(np.ceil(len(val_set) / args.batch_size))
    accum_steps = max(round(64 / args.batch_size), 1)
    global_step = tf.Variable(1, trainable=False, dtype=tf.int64)

    # print train info
    print('-' * 10)
    print('batch_size : {}'.format(args.batch_size))
    print('accum_steps : {}'.format(accum_steps))
    steps_per_epoch = int(np.ceil(len(train_set) / args.batch_size))
    if args.train_by_steps:
        print('warmup by steps : {}'.format(args.warmup_by_steps))
        if args.warmup_by_steps:
            warmup_steps = int(np.ceil(args.warmup_steps * 64 /args.batch_size))
        else:
            warmup_steps = args.warmup_epochs * steps_per_epoch
            print('warmup epochs : {}'.format(args.warmup_epochs))
        if not warmup_steps:
            warmup_steps = 1
        print('warmup steps : {}'.format(warmup_steps))
        print('train by steps : {}'.format(args.train_by_steps))
        total_steps = int(np.ceil(args.train_steps * 64 /args.batch_size))+warmup_steps
    else:
        print('warmup by steps : {}'.format(args.warmup_by_steps))
        if args.warmup_by_steps:
            warmup_steps = int(np.ceil(args.warmup_steps * 64 /args.batch_size))
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

    train_set = tf.data.Dataset.from_generator(train_set,(tf.float32,tf.float32))
    train_set = train_set.batch(args.batch_size).repeat(1)

    val_set = tf.data.Dataset.from_generator(val_set,(tf.float32,tf.float32))
    val_set=val_set.batch(args.batch_size).repeat(1)

    if args.is_tiny:
        YOLO = YOLOv4.YOLOv4_tiny(args,hyp)
    else:
        YOLO = YOLOv4.YOLOv4(args, hyp)
    
    # load pretrained model
    if args.weight_path!='':
        print('load_model from {}'.format(args.weight_path))
        YOLO.model.load_weights(args.weight_path)

    if not os.path.exists(args.weight_save_path):
        os.makedirs(args.weight_save_path)

    YOLO.model.compile(optimizer=tfa.optimizers.SGDW(learning_rate= hyp['lr0'] / warmup_steps, weight_decay=hyp['weight_decay'],
                                                     momentum=hyp['momentum']/warmup_steps,nesterov=True))

    if args.summary:
        print('Summary will be stored in {} \n You can check the learning contents with the command "tensorboard --logdir {}"'.format(args.log_path,args.log_path))
        if not os.path.exists(args.log_path):
            os.makedirs(args.log_path)
        writer = tf.summary.create_file_writer(args.log_path)
    print('-' * 10)

    #train function
    @tf.function
    def train_step(images, labels, YOLO, accum_gradient):
        with tf.GradientTape() as tape:
            pred = YOLO.model(images)
            if args.summary:
                loss_val = YOLO.loss(labels, pred,step = global_step,writer=writer)
                with writer.as_default():
                    tf.summary.scalar("learning rate", YOLO.model.optimizer.lr, step=global_step)
                    # if args.summary_variable:
                    #     for var in YOLO.model.trainable_variables[-50:]:
                    #         tf.summary.histogram(var.name, var, step=global_step)
            else:
                loss_val = YOLO.loss(labels, pred)

        # accum gradient
        gradients = tape.gradient(loss_val, YOLO.model.trainable_variables)
        accum_gradient = [(acum_grad+grad) for acum_grad,grad in zip(accum_gradient,gradients)]

        # summary gradinent histogram
        if args.summary_variable:
            with writer.as_default():
                for grad in gradients[-50:]:
                    tf.summary.histogram(grad.name, grad, step=global_step)

        # apply accum gradient
        if global_step % accum_steps == 0 or (global_step % steps_per_epoch == 0 and not args.train_by_steps and not args.warmup_by_steps):
            accum_gradient = [this_grad for this_grad in accum_gradient]
            YOLO.model.optimizer.apply_gradients(zip(accum_gradient, YOLO.model.trainable_variables))

        # learning rate schedule
        global_step.assign_add(1)
        if global_step <= warmup_steps:
            lr = global_step / warmup_steps * hyp['lr0']
            moment = global_step / warmup_steps * hyp['momentum']
            YOLO.model.optimizer.momentum.assign(tf.cast(moment,tf.float32))
        else:
            lr = hyp['lrf'] + 0.5 * ( hyp['lr0'] - hyp['lrf']) * (
                (1 + tf.cos((global_step - warmup_steps) / (total_steps - warmup_steps) * np.pi))
            )
        YOLO.model.optimizer.lr.assign(tf.cast(lr, tf.float32))
        return loss_val

    # test_function
    @tf.function
    def test_step(images, labels,YOLO):
        pred = YOLO.model(images)
        loss_val = YOLO.loss(labels, pred)
        if args.summary:
            with writer.as_default():
                tf.summary.scalar("val_loss", loss_val, step=global_step)
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

    for epoch in range(total_epochs):
        # accum gradient init
        if (global_step.numpy()-1) % accum_steps==0 or (not args.train_by_steps and not args.warmup_by_steps and (global_step.numpy()-1) % steps_per_epoch == 0):
            accum_gradient = [tf.zeros_like(this_var) for this_var in YOLO.model.trainable_variables]
            gc.collect()

        # train step
        for images, labels in train_set:
            if global_step.numpy() <= warmup_steps:
                YOLO.gr = global_step.numpy() / warmup_steps
            loss_val = train_step(images, labels,YOLO,accum_gradient)
            if global_step.numpy() % args.save_steps==0:
                YOLO.model.save_weights(args.weight_save_path + '/step_{}'.format(global_step.numpy()))

        # log print
        time_sofar = (time.time() - start_time) / 3600
        training_time_left = (total_steps / global_step.numpy() - 1.0) * time_sofar
        print("=> Epoch:%4d | Train STEP %4d | loss_val: %4.2f | time elapsed: %4.2f h | time left: %4.2f h " % (epoch,global_step.numpy(), loss_val,time_sofar,training_time_left))

        # validation step
        validation_loss = 0
        for val_images, val_labels in val_set:
            validation_loss += test_step(val_images, val_labels,YOLO)

        # log print
        time_sofar = (time.time() - start_time) / 3600
        training_time_left = (total_steps / global_step.numpy() - 1.0) * time_sofar
        print("=> Epoch:%4d | VAL STEP %4d | loss_val: %4.2f | time elapsed: %4.2f h | time left: %4.2f h " % (
        epoch, global_step.numpy(), validation_loss/val_len, time_sofar, training_time_left))

        # exit condition
        if global_step.numpy()==total_steps:
            break

        # save weight every epochs
        YOLO.model.save_weights(args.weight_save_path + '/{}'.format(epoch))

    # finish train
    YOLO.model.save_weights(args.weight_save_path+'/final')


if __name__== '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='YOLOv4 implementation.')
    parser.add_argument('--batch_size', type=int, help = 'Size of batch size. (But, The gradient update is performed every 64/batch size.) / default : 4 / 64 (YOLO v4/ YOLOv4 tiny)',
                        default=64)
    parser.add_argument('--img_size',              type=int,   help='Size of input image. / default : 416', default=416)
    parser.add_argument('--data_root',              type=str,   help='Root path of class name file and coco_%2017.txt / default : "./data"'
                                                                     , default='./data')
    parser.add_argument('--class_file',              type=str,   help='Class name file / default : "coco.name"', default='coco.names')
    parser.add_argument('--num_classes', type=int, help='Number of classes (in COCO 80) / default : 80', default=80)
    parser.add_argument('--augment',              action='store_false',   help='Flag of augmentation (hsv, flip, random affine) / default : true')
    parser.add_argument('--mosaic', action='store_false', help='Flag of mosaic augmentation / default : true')
    parser.add_argument('--is_shuffle', action='store_false', help='Flag of data shuffle / default : true')
    parser.add_argument('--train_by_steps',action='store_true', help = 'Flag for whether to proceed with the training by step or epoch. / default : false')
    parser.add_argument('--train_steps', type=int, default=500500, help ='This is the total iteration to be carried out based on 64 batches. So the total steps will be train_steps * 64 / batch_size. Used only when the train_by_steps flag is true.  / default : 500500')
    parser.add_argument('--epochs', type=int,default=300 , help ='Total epochs to be trained. Used only when the train_by_steps flag is false. / default : 300')
    parser.add_argument('--warmup_by_steps', action='store_true', help ='Flag for whether to proceed with the warm up by step or epoch. / default : false')
    parser.add_argument('--warmup_steps', type = int, default=1000, help = 'This is the total iteration of warm up to be carried out based on 64 batches. So the steps will be warm up_steps * 64 / batch_size. Used only when the warmup_by_steps flag is true. / default : 1000')
    parser.add_argument('--warmup_epochs', type=int, default=3, help ='Total epochs to warm up. Used only when the warmup_by_steps flag is false. / default : 3')
    parser.add_argument('--save_steps', type=int, default=1000, help ='Step cycle to store weight. /default : 1000')
    parser.add_argument('--is_tiny', action='store_true', help ='Flag for using tiny. / default : false')
    parser.add_argument('--soft',type=float,default=0.0, help = 'This is a value for soft labeling, and soft/num_class becomes the label for negative class. / default : 0.0' )
    parser.add_argument('--log_path', type=str, default='./log/tiny', help = 'logdir path for Tensorboard')
    parser.add_argument('--summary', action='store_false')
    parser.add_argument('--summary_variable', action='store_false')
    parser.add_argument('--weight_path',type=str,default='',help = 'Path of weight file / default : ""')
    parser.add_argument('--weight_save_path', type=str, default='./weight',help = 'Path to store weights. / default : "./wegiht"')
    parser.add_argument('--mode',
                        default='train',
                        const='train',
                        nargs='?',
                        choices=['train', 'test'],
                        help='Mode [train, test]')
    args = parser.parse_args()
    hyp = {'giou': 0.05,#3.54,  # giou loss gain
           'cls': 0.5,#37.4,  # cls loss gain
           'obj': 1.0,#83.59,  # obj loss gain (=64.3*img_size/320 if img_size != 320)
           'iou_t': 0.213,  # iou training threshold
           'lr0': 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)
           'lrf': 0.0013,  # final learning rate (with cos scheduler)
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

    train(args,hyp)