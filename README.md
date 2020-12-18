# YOLO v4
YOLOv4, YOLOv4-tiny Implemented in Tensorflow 2.0. 
<br>

## COCO Dataset to YOLO format

- Please refer to this [url](https://github.com/Songminkee/YOLOv4_keras_implementation/blob/master/convert_util/convert_command.md).

```
data_root_path(default:./data)
ㄴdataset
	ㄴcoco_val2017.txt
	ㄴcoco_train2017.txt
```

```
in coco_%2017.txt
dataset_folder_path/images/%2017/000000000139.jpg
.....
```

```
dataset_folder_path
ㄴimages
	ㄴval2017
		ㄴ000000000139.jpg
		ㄴ....jpg
	ㄴtrain2017
		ㄴ....jpg
ㄴlabels
	ㄴval2017
		ㄴ000000000139.txt
		ㄴ....txt
	ㄴtrain2017
		ㄴ....txt
```

<br>

## Train and Test(eval)

If you run the following, it will be executed as default, and the description of additional parameters is as follows.

```bash
python train.py
python eval.py
```
<br>

## Train argument

- `batch_size` : Size of batch size. (But, The gradient update is performed every 64/batch size.) / default : 4 / 64 (YOLO v4/ YOLOv4 tiny)
- `img_size` : Size of input image. / default : 416
- `data_root`: Root path of class name file and coco_%2017.txt / default : './data'
- `class_file` : Class name file / default : 'coco.name'
- `num_classes` : Number of classes (in COCO 80) / default : 80
- `augment` : Flag of augmentation (hsv, flip, random affine) / default : true
- `mosaic` : Flag of mosaic augmentation / default : true
- `is_shuffle` : Flag of data shuffle / default : true
- `train_by_steps` : Flag for whether to proceed with the training by step or epoch. / default : false
- `train_steps` : This is the total iteration to be carried out based on 64 batches. So the total steps will be train_steps * 64 / batch_size. Used only when the train_by_steps flag is true.  / default : 500500

- `epochs` : Total epochs to be trained. Used only when the train_by_steps flag is false. / default : 300
- `warmup_by_steps` : Flag for whether to proceed with the warm up by step or epoch. / default : false
- `warmup_steps` : This is the total iteration of warm up to be carried out based on 64 batches. So the steps will be warm up_steps * 64 / batch_size. Used only when the warmup_by_steps flag is true. / default : 1000
- `warmup_epochs` : Total epochs to warm up. Used only when the warmup_by_steps flag is false. / default : 3
- `save_steps` : Step cycle to store weight. /default : 1000
- `is_tiny` : Flag for using tiny. / default : false
- `soft` : This is a value for soft labeling, and soft/num_class becomes the label for negative class. / default : 0.0

- `log_path` : logdir path for Tensorboard
- `weight_path` : Path of weight file / default : ''

- `weight_save_path` : Path to store weights. / default : './wegiht'

<br>

## Test argument

- `img_size` : Size of input image. / default : 416
- `data_root`: Root path of class name file and coco_%2017.txt / default : './data'
- `class_file` : Class name file / default : 'coco.name'
- `num_classes` : Number of classes (in COCO 80) / default : 80

- `augment` : Flag of augmentation (hsv, flip, random affine) / default : false
- `mosaic` : Flag of mosaic augmentation / default : false

- `is_shuffle` : Flag of data shuffle / default : true

- `only_coco_eval` : When the flag is true, only pycocotools is used to show the result. In this case, you must enter the path to the json file. / default : false
- `out_json_path` : Folder of output file (json) / default : './eval'
- `json_path` : Path of json result file. This flag only used when the only_coco_eval flag is true
- `weight_path` : path of weight
- `annotation_path` : COCO annotation file folder / default : './data/dataset/COCO/annotations'
- `is_darknet_wegiht` : If true, load the weight file used by the darknet framework.
- `is_tiny` : Flag for using tiny. / default : false

- `is_padding` : If true, padding is performed to maintain the ratio of the input image. / default : false



# Convert to tflite

1. Download weights from [darknet](https://github.com/AlexeyAB/darknet) or [yolov4.weights](https://drive.google.com/file/d/1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT/view)

2. Convert to keras model

   ```
   python save_model.py --data_root ./data --class_file coco.names --is_darknet_weight --weight_path ./weight/yolov4.weights --num_classes 80 --out_path ./saved_model/darknet_weight
   ```

3. Convert to tflite 

   ```
   python convert_tflite.py --num_classes 80 --dataset './data/dataset/coco_val2017.txt' --weight_path ./saved_model/darknet_weight --out_model ./darknet.tflite --quantize_mode float16
   ```

4. Detect

   ```
   python detect.py --num_classes 80 --input_dir './data/dataset/COCO/images/val2017' --weight_path darknet.tflite --data_name cooc --is_tflite
   ```

   



### References

  * YOLOv4: Optimal Speed and Accuracy of Object Detection [YOLOv4](https://arxiv.org/abs/2004.10934).
  * [darknet](https://github.com/AlexeyAB/darknet)
  * [Pytorch YOLOv4 Implemenation](https://github.com/WongKinYiu/PyTorch_YOLOv4)

- [tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite)

