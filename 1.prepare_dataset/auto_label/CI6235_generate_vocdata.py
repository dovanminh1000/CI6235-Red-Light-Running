#CI6235: Reference from https://github.com/squeakus/bitsandbytes/blob/master/pytorchtools/generate_vocdata.py
#CI6235: Divide the data into 80% for training, 10% for evaluation and the rest 10% for testing

import glob
import sys
import os
import xml.etree.ElementTree as ET
from random import random



def main(filename, dataset):
    # ratio to divide up the images
    targetDir = "dataset_generation/dataset/" + dataset + "/"
    IMAGE_PATH = targetDir
    ANNO_PATH = targetDir

    WRITE_PATH = "dataset_generation/split_sets/" + dataset
    if not os.path.exists("../" + WRITE_PATH):
        os.mkdir("../" + WRITE_PATH)
    train = 0.8
    val = 0.1
    test = 0.1
    if (train + test + val) != 1.0:
        print ("probabilities must equal 1")
        exit()

    # get the labels
    labels = []
    imgnames = []
    annotations = {}

    with open(filename, "r") as labelfile:
        label_string = ""
        for line in labelfile:
            label_string += line.rstrip()

    labels = label_string.split(",")
    labels = [elem.replace(" ", "") for elem in labels]

    # get image names
    for filename in os.listdir("../" + targetDir):
        if filename.endswith(".jpg"):
            img = filename.rstrip(".jpg")
            imgnames.append(img)

    print ("Labels:", labels, "imgcnt:", len(imgnames))

    # initialise annotation list
    for label in labels:
        annotations[label] = []

    # Scan the annotations for the labels
    for img in imgnames:
        annote = "../" + targetDir + img + ".xml"
        if os.path.isfile(annote):
            print annote
            tree = ET.parse(annote)
            root = tree.getroot()
            annote_labels = []
            for labelname in root.findall("*/name"):
                labelname = labelname.text
                annote_labels.append(labelname)
                if labelname in labels:
                    annotations[labelname].append(img)
            annotations[img] = annote_labels
        else:
            print ("Missing annotation for ", annote)
            exit()

    # divvy up the images to the different sets
    sampler = imgnames[:]
    train_list = []
    val_list = []
    test_list = []

    while len(sampler) > 0:
        dice = random()
        elem = sampler.pop()

        if dice <= test:
            test_list.append(elem)
        elif dice <= (test + val):
            val_list.append(elem)
        else:
            train_list.append(elem)

    print (
        "Training set:",
        len(train_list),
        "validation set:",
        len(val_list),
        "test set:",
        len(test_list),
    )

    # create the dataset files
    # create_folder("../" + WRITE_PATH+ "")
    with open("../" + WRITE_PATH + "/train.txt", "w") as outfile:
        for name in train_list:
            outfile.write(name + ".jpg " + name + ".xml\n")
    with open("../" + WRITE_PATH + "/train_tf.txt", "w") as outfile:
        for name in train_list:
            outfile.write(name + "\n")
    with open("../" + WRITE_PATH + "/val.txt", "w") as outfile:
        for name in val_list:
            outfile.write(name + ".jpg " + name + ".xml\n")
    with open("../" + WRITE_PATH + "/val_tf.txt", "w") as outfile:
        for name in val_list:
            outfile.write(name + "\n")
    with open("../" + WRITE_PATH + "/trainval.txt", "w") as outfile:
        for name in train_list:
            outfile.write(name + ".jpg " + name + ".xml\n")
        for name in val_list:
            outfile.write(name + ".jpg " + name + ".xml\n")

    with open("../" + WRITE_PATH + "/test.txt", "w") as outfile:
        for name in test_list:
            outfile.write(name + ".jpg " + name + ".xml\n")
    with open("../" + WRITE_PATH + "/test_tf.txt", "w") as outfile:
        for name in test_list:
            outfile.write(name + "\n")

    # create the individiual files for each label
    for label in labels:
        with open("../" + WRITE_PATH + "/" + label + "_train.txt", "w") as outfile:
            for name in train_list:
                if label in annotations[name]:
                    outfile.write(name + " 1\n")
                else:
                    outfile.write(name + " -1\n")
        with open("../" + WRITE_PATH + "/" + label + "_val.txt", "w") as outfile:
            for name in val_list:
                if label in annotations[name]:
                    outfile.write(name + " 1\n")
                else:
                    outfile.write(name + " -1\n")
        with open("../" + WRITE_PATH + "/" + label + "_test.txt", "w") as outfile:
            for name in test_list:
                if label in annotations[name]:
                    outfile.write(name + " 1\n")
                else:
                    outfile.write(name + " -1\n")


def create_folder(foldername):
    if os.path.exists(foldername):
        print ("folder already exists:", foldername)
    else:
        os.makedirs(foldername)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2:
        main(sys.argv[1], "CI6235")
    else:
        print (
            "usage: python generate_vocdata.py <labelfile> <dataset_name (default = CI6235)>"
        )
        exit()
