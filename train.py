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

print(tf.config.experimental.list_physical_devices('GPU'))

def train(args,hyp):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    train_set = loader.DataLoader(args, hyp,'train')
    val_set = loader.DataLoader(args, hyp, 'val')
    accum_steps = max(round(64 / args.batch_size), 1)
    steps_per_epoch = int(np.ceil(len(train_set) / args.batch_size))
    global_step = tf.Variable(1, trainable=False, dtype=tf.int64)
    warmup_steps = args.warmup_epochs * steps_per_epoch
    total_steps = (args.warmup_epochs+args.epochs) * steps_per_epoch

    train_set = tf.data.Dataset.from_generator(train_set,(tf.float32,tf.float32))
    train_set = train_set.batch(args.batch_size).repeat(1)

    burn = max(3 * steps_per_epoch, 500)

    val_set = tf.data.Dataset.from_generator(val_set,(tf.float32,tf.float32))
    val_set=val_set.batch(args.batch_size).repeat(1)

    YOLO = YOLOv4.YOLOv4(args, hyp)
    if args.weight_path!='':
        print('load_model from {}'.format(args.weight_path))
        YOLO.model.load_weights(args.weight_path)

    if not os.path.exists(args.weight_save_path):
        os.makedirs(args.weight_save_path)

    YOLO.model.compile(optimizer=tfa.optimizers.SGDW(learning_rate= hyp['lr0'] / warmup_steps, weight_decay=hyp['weight_decay'],
                                                     momentum=hyp['momentum']/warmup_steps,nesterov=True))

    if args.summary:
        print('Summary will be stored in {} \n You can check the learning contents with the command "tensorboard --logdir {}"'.format(args.log_path,args.log_path))
        writer = tf.summary.create_file_writer(args.log_path)

    @tf.function
    def train_step(images, labels, YOLO, accum_gradient):
        with tf.GradientTape() as tape:
            pred = YOLO.model(images)
            if args.summary:
                loss_val = YOLO.loss(labels, pred,step = global_step,writer=writer)
                with writer.as_default():
                    tf.summary.scalar("learning rate", YOLO.model.optimizer.lr, step=global_step)
                    if args.summary_variable:
                        for var in YOLO.model.trainable_variables[-50:]:
                            tf.summary.histogram(var.name, var, step=global_step)
            else:
                loss_val = YOLO.loss(labels, pred)

        gradients = tape.gradient(loss_val, YOLO.model.trainable_variables)
        accum_gradient = [(acum_grad+grad) for acum_grad,grad in zip(accum_gradient,gradients)]

        if args.summary_variable:
            with writer.as_default():
                for grad in gradients[-50:]:
                    tf.summary.histogram(grad.name, grad, step=global_step)
        YOLO.model.optimizer.apply_gradients(zip(gradients, YOLO.model.trainable_variables))

        if global_step % accum_steps == 0 or global_step % steps_per_epoch == 0:
            accum_gradient = [this_grad for this_grad in accum_gradient]
            YOLO.model.optimizer.apply_gradients(zip(accum_gradient, YOLO.model.trainable_variables))

        # learning rate schedule
        global_step.assign_add(1)
        if global_step < warmup_steps:
            lr = global_step / warmup_steps * hyp['lr0']
            moment = global_step / warmup_steps * hyp['momentum']
            YOLO.model.optimizer.momentum.assign(tf.cast(moment,tf.float32))

        else:
            lr = hyp['lrf'] + 0.5 * ( hyp['lr0'] - hyp['lrf']) * (
                (1 + tf.cos((global_step - warmup_steps) / (total_steps - warmup_steps) * np.pi))
            )
        YOLO.model.optimizer.lr.assign(tf.cast(lr, tf.float32))
        return loss_val

    @tf.function
    def test_step(images, labels,YOLO):
        pred = YOLO.model(images)
        loss_val = YOLO.loss(labels, pred)
        if args.summary:
            with writer.as_default():
                tf.summary.scalar("val_loss", loss_val, step=global_step)
        return loss_val

    train_vars = YOLO.model.trainable_variables
    accum_gradient = [tf.zeros_like(this_var) for this_var in train_vars]
    start_time = time.time()
    for epoch in range(args.warmup_epochs+args.epochs):
        if (global_step.numpy()-1) % accum_steps==0:
            train_vars = YOLO.model.trainable_variables
            accum_gradient = [tf.zeros_like(this_var) for this_var in train_vars]

        # train step
        for images, labels in train_set:
            if global_step.numpy() <= warmup_steps:
                YOLO.gr = global_step.numy() / warmup_steps
            loss_val = train_step(images, labels,YOLO,accum_gradient)
            if global_step.numpy() % 500==0:
                YOLO.model.save_weights(args.weight_save_path + '/step_{}'.format(global_step.numpy()))

        time_sofar = (time.time() - start_time) / 3600
        training_time_left = (total_steps / global_step.numpy() - 1.0) * time_sofar

        print("=> Epoch:%4d | Train STEP %4d | loss_val: %4.2f | time elapsed: %4.2f h | time left: %4.2f h " % (epoch,global_step.numpy(), loss_val,time_sofar,training_time_left))

        # validation step
        for val_images, val_labels in val_set:
            loss_val = test_step(val_images, val_labels,YOLO)
            break

        time_sofar = (time.time() - start_time) / 3600
        training_time_left = (total_steps / global_step.numpy() - 1.0) * time_sofar

        print("=> Epoch:%4d | VAL STEP %4d | loss_val: %4.2f | time elapsed: %4.2f h | time left: %4.2f h " % (
        epoch, global_step.numpy(), loss_val, time_sofar, training_time_left))
        YOLO.model.save_weights(args.weight_save_path + '/{}'.format(epoch))

    YOLO.model.save_weights(args.weight_save_path+'/final')


if __name__== '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Darknet53 implementation.')
    parser.add_argument('--batch_size', type=int, help = 'size of batch', default=2)
    parser.add_argument('--img_size',              type=int,   help='input height', default=512)
    parser.add_argument('--data_root',              type=str,   help='', default='./data')
    parser.add_argument('--class_file',              type=str,   help='', default='coco.names')
    parser.add_argument('--num_classes', type=int, help='', default=80)
    parser.add_argument('--augment',              action='store_false',   help='')
    parser.add_argument('--mosaic', action='store_false', help='')
    parser.add_argument('--is_shuffle', action='store_false', help='')
    parser.add_argument('--epochs', type=int,default=300 )
    parser.add_argument('--warmup_epochs', type=int, default=2)
    parser.add_argument('--soft',type=float,default=0.0)
    parser.add_argument('--log_path', type=str, default='./log/burn')
    parser.add_argument('--summary', action='store_false')
    parser.add_argument('--summary_variable', action='store_false')
    parser.add_argument('--weight_path',type=str,default='')
    parser.add_argument('--weight_save_path', type=str, default='./weight')
    parser.add_argument('--mode',
                        default='train',
                        const='train',
                        nargs='?',
                        choices=['train', 'test'],
                        help='Mode [train, test]')
    args = parser.parse_args()
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

    train(args,hyp)