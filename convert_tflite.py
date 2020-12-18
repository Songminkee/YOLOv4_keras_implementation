import tensorflow as tf
import numpy as np
import cv2
import os

import argparse

parser = argparse.ArgumentParser(description='YOLOv4 Test')
parser.add_argument('--img_size', type=int, help='Size of input image / default : 416', default=416)
parser.add_argument('--dataset', type=str,
                    help='Root path of coco_%2017.txt / default : "./data"', default='./data/dataset/coco_val2017.txt')
parser.add_argument('--num_classes', type=int, help='Number of classes (in COCO 80) ', default=80)  # 80
parser.add_argument('--weight_path', type=str, default='./saved_model/fine_tune',
                    help='path of weight')
parser.add_argument('--out_model',type=str, default='./fine_tune16.tflite',
                    help='name of tflite model')
parser.add_argument('--quantize_mode', type=str, default='float16',
                    help='quantize_mode of tflite model')
parser.add_argument('--is_padding', action='store_true',
                    help=' If true, padding is performed to maintain the ratio of the input image. / default : false')
args = parser.parse_args()
def letterbox(img, new_shape=(416, 416), color=(114, 114, 114)):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img

def representative_data_gen():
    fimage = open(args.dataset).read().split()
    for input_value in range(10):
        if os.path.exists(fimage[input_value]):
            original_image = cv2.imread(fimage[input_value])
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            image_data = cv2.resize(np.copy(original_image), (args.img_size, args.img_size)) / 255.

            img_in = image_data[np.newaxis, ...].astype(np.float32)
            print(img_in.shape)
            print("calibration image {}".format(fimage[input_value]))
            yield [img_in]
        else:
            continue

def save_tflite(args):
    converter = tf.lite.TFLiteConverter.from_saved_model(args.weight_path)

    print('-'*30)
    if args.quantize_mode == 'float16':
        print("convert to full float16")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.compat.v1.lite.constants.FLOAT16]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        converter.allow_custom_ops = True
    elif args.quantize_mode == 'full_int8':
        print("convert to full int8") # There are bugs depending on the version. It will be revised later.
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
        converter.allow_custom_ops = True
        converter.representative_dataset = representative_data_gen
    else:
        print(f"{args.quantize_mode} not provided")
        exit()


    tflite_model = converter.convert()

    open(args.out_model, 'wb').write(tflite_model)
    print("Write outmodel")
    print('-' * 30)

def demo(args):
    print("start demo")
    interpreter = tf.lite.Interpreter(model_path=args.out_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    print(input_details)
    output_details = interpreter.get_output_details()
    print(output_details)

    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

    print("demo complete")


if __name__ == '__main__':

    save_tflite(args)
    demo(args) # for error check
