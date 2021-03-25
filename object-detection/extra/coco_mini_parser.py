import ijson
import shutil
import os
import sys
from argparse import ArgumentParser

def main(args):
  ANNOTATION_DIR = args.annotation_dir
  COCO_DATASET_DIR = args.coco_dataset_dir
  DESTINATION_DIR = args.destination_dir

  data = ijson.parse(open(ANNOTATION_DIR, 'r'))

  file_names = []
  for prefix, event, value in data:
    if prefix == "images.item.file_name":
      file_names.append(value)
      
  for file in file_names:
    srcpath = os.path.join(COCO_DATASET_DIR, file)
    if not os.path.isdir(DESTINATION_DIR):
        os.makedirs(DESTINATION_DIR)
    dstpath = os.path.join(DESTINATION_DIR, file)
    shutil.copyfile(srcpath, dstpath)

if __name__ == '__main__':
  parser = ArgumentParser('python3 coco_mini_parser.py')

  parser.add_argument('--annotation_dir', type=str)
  parser.add_argument('--coco_dataset_dir', type=str)
  parser.add_argument('--destination_dir', type=str)

  main(parser.parse_args())
  