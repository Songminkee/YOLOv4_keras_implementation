# -*-coding:utf-8-*-

import os
from xml.etree.ElementTree import dump
import json
import pprint
import glob
import argparse


from Format import COCO,YOLO

parser = argparse.ArgumentParser(description='label Converting example.')

parser.add_argument('--img_path', type=str, help='directory of image folder')
parser.add_argument('--label', type=str,
                    help='directory of label folder or label file path',default="")
parser.add_argument('--convert_output_path', type=str,
                    help='directory of label folder', default="./label")
parser.add_argument('--img_type', type=str, help='type of image')
parser.add_argument('--manifest_path', type=str,
                    help='directory of manipast file', default="./")
parser.add_argument('--cls_list_file', type=str,
                    help='directory of *.names file', default="./")
parser.add_argument('--manifest_name', type=str,
                    help='name of manipast file', default="manifest")
parser.add_argument('--no_label', action='store_true',
                    help='name of manipast file')

args = parser.parse_args()

def make_image_path(config):
    paths =  glob.glob(os.path.join(config["img_path"],'*'+config["img_type"]))
    with open(config["manifest_path"]+config["manifest_name"]+'.txt','w') as f:
        for i in range(len(paths)):
            f.write(paths[i]+'\n')


def main(config):
    if config["no_label"]:
        make_image_path(config)
        return
    coco = COCO()
    print(config["label"])
    print(config["img_path"])
    flag, data, cls_hierarchy = coco.parse(
        config["label"], config["img_path"])
    yolo = YOLO(os.path.abspath(
        config["cls_list"]), cls_hierarchy=cls_hierarchy)

    if flag == True:
        flag, data = yolo.generate(data)

        if flag == True:
            flag, data = yolo.save(data, config["output_path"], config["img_path"],
                                   config["img_type"], config["manifest_path"],config["manifest_name"])

            if flag == False:
                print("Saving Result : {}, msg : {}".format(flag, data))
        else:
            print("YOLO Generating Result : {}, msg : {}".format(flag, data))
    else:
        print("COCO Parsing Result : {}, msg : {}".format(flag, data))


if __name__ == '__main__':

    config = {
        "img_path": args.img_path,
        "label": args.label,
        "img_type": args.img_type,
        "manifest_path": args.manifest_path,
        "output_path": args.convert_output_path,
        "cls_list": args.cls_list_file,
        "manifest_name" : args.manifest_name,
        "no_label" : args.no_label
    }

    main(config)