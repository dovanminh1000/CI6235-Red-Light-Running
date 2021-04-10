#CI6235: Reference: https://gist.github.com/goodhamgupta/7ca514458d24af980669b8b1c8bcdafd
#CI6235: Convert yolo annotation to voc format 

# Script to convert yolo annotations to voc format
# Sample format
# <annotation>
#     <folder>_image_fashion</folder>
#     <filename>brooke-cagle-39574.jpg</filename>
#     <size>
#         <width>1200</width>
#         <height>800</height>
#         <depth>3</depth>
#     </size>
#     <segmented>0</segmented>
#     <object>
#         <name>head</name>
#         <pose>Unspecified</pose>
#         <truncated>0</truncated>
#         <difficult>0</difficult>
#         <bndbox>
#             <xmin>549</xmin>
#             <ymin>251</ymin>
#             <xmax>625</xmax>
#             <ymax>335</ymax>
#         </bndbox>
#     </object>
# <annotation>
import os
import xml.etree.cElementTree as ET
from PIL import Image
import sys

if len(sys.argv) == 2:
    dataset = sys.argv[1]
elif len(sys.argv) == 1:
    dataset = "CI6235"
else:
    print("usage: python yolo_pvoc.py <dataset_name (default = CI6235)>")
    exit()
TARGET_DIR = "../dataset_generation/dataset/" + dataset + "/"

CLASS_MAPPING = {
    "0": "vehicle"
    # Add your remaining classes here.
}


def create_root(file_prefix, width, height):
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = "{}.jpg".format(file_prefix)
    ET.SubElement(root, "folder").text = "images"
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    return root


def create_object_annotation(root, voc_labels):
    for voc_label in voc_labels:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = voc_label[0]
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = str(0)
        ET.SubElement(obj, "difficult").text = str(0)
        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(max(voc_label[1], 0))
        ET.SubElement(bbox, "ymin").text = str(max(voc_label[2], 0))
        ET.SubElement(bbox, "xmax").text = str(min(voc_label[3], 1920))
        ET.SubElement(bbox, "ymax").text = str(min(voc_label[4], 1080))
    return root


def create_file(file_prefix, width, height, voc_labels):
    root = create_root(file_prefix, width, height)
    root = create_object_annotation(root, voc_labels)
    tree = ET.ElementTree(root)
    tree.write("{}/{}.xml".format(TARGET_DIR, file_prefix))


def read_file(file_path):
    file_prefix = file_path.split(".txt")[0]
    image_file_name = "{}.jpg".format(file_prefix)
    try:
        img = Image.open("{}/{}".format(TARGET_DIR, image_file_name))
    except IOError:
        os.remove(TARGET_DIR + "/" + file_path)
        return
    w, h = img.size
    with open(TARGET_DIR + "/" + file_path, "r") as file:
        lines = file.readlines()
        voc_labels = []
        for line in lines:
            voc = []
            line = line.strip()
            data = line.split()
            voc.append(CLASS_MAPPING.get(data[0]))
            bbox_width = int(float(data[3]) * w)
            bbox_height = int(float(data[4]) * h)
            center_x = int(float(data[1]) * w)
            center_y = int(float(data[2]) * h)
            voc.append(center_x - (bbox_width / 2))
            voc.append(center_y - (bbox_height / 2))
            voc.append(center_x + (bbox_width / 2))
            voc.append(center_y + (bbox_height / 2))
            voc_labels.append(voc)
        create_file(file_prefix, w, h, voc_labels)
    print("Processing complete for file: {}".format(file_path))


def start(dir_name):
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
    for filename in os.listdir(TARGET_DIR):
        if filename.endswith("txt"):
            read_file(filename)
        else:
            print("Skipping file: {}".format(filename))


if __name__ == "__main__":
    start("")
