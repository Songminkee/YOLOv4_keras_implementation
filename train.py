import os
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow_addons as tfa
from model import YOLOv4
from dataloader import loader
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard


print(tf.config.experimental.list_physical_devices('GPU'))
class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir, **kwargs):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)

@tf.function
def train_step(images, labels,model):
    with tf.GradientTape() as tape:
        pred = model(images)
        loss_val = model.loss(labels,pred)
    gradients = tape.gradient(loss_val, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))



def train(args,hyp):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    YOLO = YOLOv4.YOLOv4(args,hyp)

    train_set = loader.DataLoader(args, hyp,'train')
    steps_per_epoch = len(train_set)
    train_set = tf.data.Dataset.from_generator(loader.DataLoader(args, hyp,'train'),(tf.float32,tf.float32))
    val_set = tf.data.Dataset.from_generator(loader.DataLoader(args, hyp,'val'),(tf.float32,tf.float32))
    train_set=train_set.batch(args.batch_size).repeat(1)
    val_set=val_set.batch(args.batch_size).repeat(1)

    global_step = tf.Variable(1, trainable = False, dtype=tf.int64)
    warmup_steps = args.warmup_epoch * steps_per_epoch

    EPOCHS = 5
    YOLO.model.compile(optimizer=tfa.optimizers.SGDW(learning_rate=hyp['lr0'], weight_decay=hyp['weight_decay'],
                                                     momentum=hyp['momentum']))
    # TODO:
    # for epoch in range(EPOCHS):
    #     for images, labels in train_set:
    #         train_step(images, labels)
    # 
    #     for test_images, test_labels in val_set:
    #         test_step(test_images, test_labels)



    YOLO.model.fit(train_set,batch_size=4,epochs=1,validation_data=val_set)

    # cnt=0
    # for img,label in val_set.as_numpy_iterator():
    #     out = YOLO.model(img)
    #     loss = YOLO.loss(label,out)
    #     cnt+=1
    #     if cnt==2:
    #         break
    #
    # for img,label in val_set.as_numpy_iterator():
    #     out = YOLO.model(img)
    #     ret = YOLO.loss(label, out)

    #YOLO.model.optimizer.lr.assign()

if __name__== '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Darknet53 implementation.')
    parser.add_argument('--batch_size', type=int, help = 'size of batch', default=4)
    parser.add_argument('--img_size',              type=int,   help='input height', default=512)
    parser.add_argument('--data_root',              type=str,   help='', default='./data')
    parser.add_argument('--class_file',              type=str,   help='', default='coco.names')
    parser.add_argument('--num_classes', type=int, help='', default=80)
    parser.add_argument('--augment',              action='store_false',   help='')
    parser.add_argument('--mosaic', action='store_false', help='')
    parser.add_argument('--is_shuffle', action='store_false', help='')
    parser.add_argument('--epoch', type=int,default=30 )
    parser.add_argument('--warmup_epoch', type=int, default=2)
    parser.add_argument('--soft',type=float,default=0.0)
    parser.add_argument('--mode',
                        default='train',
                        const='train',
                        nargs='?',
                        choices=['train', 'test'],
                        help='Mode [train, test]')
    args = parser.parse_args()
    hyp = {'giou': 3.54,  # giou loss gain
           'cls': 37.4,  # cls loss gain
           'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
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