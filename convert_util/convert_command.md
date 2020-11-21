# Convert COCO format to YOLO format

```
python convert_util/converter.py --img_path [coco image folder] --label [coco annotations file] --convert_output_path [label file output folder] --img_type ".jpg" --manifest_path [manifest folder] --cls_list_file [coco name file] --manifest_name [manifest file name]
```



#### ex)

```
python convert_util/converter.py --img_path /mnt/e/00.dataset/COCO/images/train2017/ --label /mnt/e/00.dataset/COCO/annotations/instances_train2017.json --convert_output_path /mnt/e/00.dataset/COCO/labels/train2017 --img_type ".jpg" --manifest_path ./ --cls_list_file ./data/dataset/COCO/annotations/coco.names --manifest_name coco_train2017
```



```
python convert_util/converter.py --img_path /mnt/e/00.dataset/COCO/images/val2017/ --label /mnt/e/00.dataset/COCO/annotations/instances_val2017.json --convert_output_path /mnt/e/00.dataset/COCO/labels/val2017 --img_type ".jpg" --manifest_path ./ --cls_list_file ./data/dataset/COCO/annotations/coco.names --manifest_name coco_val2017
```



```
python convert_util/converter.py --img_path /mnt/e/00.dataset/COCO/images/test2017/ --no_label --img_type ".jpg" --manifest_path ./ --manifest_name coco_test2017
```



