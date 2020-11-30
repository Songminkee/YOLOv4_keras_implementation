# Convert COCO format to YOLO format

```
python convert_util/converter.py --img_path [coco image folder] --label [coco annotations file] --convert_output_path [label file output folder] --img_type ".jpg" --manifest_path [manifest folder] --cls_list_file [coco name file] --manifest_name [manifest file name]
```



#### ex)

```
python convert_util/converter.py --img_path ./data/dataset/COCO/images/train2017/ --label ./data/dataset/COCO/annotations/instances_train2017.json --convert_output_path ./data/dataset/COCO/labels/train2017 --img_type ".jpg" --manifest_path ./data/dataset/ --cls_list_file ./data/dataset/COCO/annotations/coco.names --manifest_name coco_train2017
```


