from ctypes import *  # Provide C capatible data types
import math
import random
import os
import sys
import cv2
from Ensemble import (  # Model collection
    Ensemble,
    GoogLeNet,
    Net,
    BasicConv2d,
    Loaders,
    Inception,
    InceptionAux,
)
#CI6235: Define input data and intialize necessar variables
data_path = "/CI6235/assignment1/final/data_preparation/dataset/RLR/auto_label/"
dataset = ""
total_remove_file = 0
total_keep_file = 0


def sample(probs):
    s = sum(probs)
    probs = [a / s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs) - 1


def c_array(ctype, values):
    arr = (ctype * len(values))()
    arr[:] = values
    return arr


class BOX(Structure):  # Bounding box coordination and dimension
    _fields_ = [("x", c_float), ("y", c_float), ("w", c_float), ("h", c_float)]


class DETECTION(Structure):  # Return format of detection
    _fields_ = [
        ("bbox", BOX),
        ("classes", c_int),
        ("prob", POINTER(c_float)),
        ("mask", POINTER(c_float)),
        ("objectness", c_float),
        ("sort_class", c_int),
    ]


class IMAGE(Structure):  # Image structure
    _fields_ = [("w", c_int), ("h", c_int), ("c", c_int), ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int), ("names", POINTER(c_char_p))]


# lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL(
    os.path.join(os.getcwd(), "libdarknet.so"), RTLD_GLOBAL
)  # Read darknet library
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [
    c_void_p,
    c_int,
    c_int,
    c_float,
    c_float,
    POINTER(c_int),
    c_int,
    POINTER(c_int),
]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

# CI6235: Draw bounding box of vehicles on each frames -> use to manually clean the result later
def drawBox(
    i, pred, image
):  # Draw bounding box of object, i: array of results, pred: prediction score, image: frame need to be drawn result
    print("i: ", i)
    print("pred: ", pred)
    colormap = {
        "bus": (0, 255, 0),
        "car": (255, 0, 0),
        "truck": (0, 0, 255),
    }
    x1 = int((i[2][0] - i[2][2] / 2))
    y1 = int((i[2][1] - i[2][3] / 2))
    x2 = int((i[2][0] + i[2][2] / 2))
    y2 = int((i[2][1] + i[2][3] / 2))
    txt = "%s %.2f" % (i[0], i[1])
    width, height = cv2.getTextSize(txt, 0, 1, 2)[0]
    cv2.rectangle(image, (x1, y1), (x2, y2), colormap[i[0]], 1)
    cv2.putText(image, txt, (x1, y1), 0, 0.5, colormap[i[0]], 1)
    return image

# CI6235: Predict bus, car and truck for the input image
def detect(net, meta, image, thresh=0.4, hier_thresh=0.5, nms=0.45, classes=1):
    global total_keep_file
    global total_remove_file
    # global out
    im = load_image(image, 0, 0)
    cv_im = cv2.imread(image)
    op_im = cv_im.copy()
    run = Loaders()
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)  # predict image
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)

    label_map = {"bus": 0, "car": 0, "truck": 0}
    num = pnum[0]
    print("num: ", num)
    if nms:
        do_nms_obj(
            dets, num, meta.classes, nms
        )  # Choose the highest score bounding boxes and remove the bounding boxes overlap with these ones

    res = []  # contain output result, which includes class name, score and coordinators
    for j in range(num):
        for i in range(meta.classes):
            if (dets[j].prob[i] > 0) and (
                meta.names[i] == "bus"
                or meta.names[i] == "car"
                or meta.names[i] == "truck"
            ):
                # input check to ensure size is sufficient
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    idx = [0, 0, 0]  #CI6235: Count number of each predicted class: bus, car, truck
    rt = 0  # whether having vehicles on image on not
    if len(res) > 0:
        with open(
            DataDir + image.split("/")[-1].split(".")[0] + ".txt", "w"
        ) as f:  #CI6235: Open txt file to write predicted objects
            for i in res:
                pred = i[0]
                if i[0] == "bus":
                    if i[2][2] >= 15 and i[2][3] >= 15:  #CI6235: Remove too small bus (bus is too far from camera)
                        f.write(
                            "%d %f %f %f %f\n"
                            % (
                                label_map[pred],
                                i[2][0] / im.w,
                                i[2][1] / im.h,
                                i[2][2] / im.w,
                                i[2][3] / im.h,
                            )
                        )
                        idx[0] += 1
                        rt = 1
                elif i[0] == "car" or i[0] == "truck":
                    if i[0] == "car":
                        if i[2][2] >= 15 and i[2][3] >= 15:   #CI6235: Remove too small car (car is too far from camera)
                            idx[1] += 1

                            f.write(
                                "%d %f %f %f %f\n"
                                % (
                                    label_map[pred],
                                    i[2][0] / im.w,
                                    i[2][1] / im.h,
                                    i[2][2] / im.w,
                                    i[2][3] / im.h,
                                )
                            )
                            rt = 1

                    if i[0] == "truck":
                        if i[2][2] >= 15 and i[2][3] >= 15:   #CI6235: Remove too small truck (truck is too far from camera)
                            idx[2] += 1
                            f.write(
                                "%d %f %f %f %f\n"
                                % (
                                    label_map[pred],
                                    i[2][0] / im.w,
                                    i[2][1] / im.h,
                                    i[2][2] / im.w,
                                    i[2][3] / im.h,
                                )
                            )
                            rt = 1
                else:
                    continue

                if rt == 1:
                    op_im = drawBox(i, pred, op_im)  #CI6235: Draw bounding box of vehicles on input image

    res = sorted(res, key=lambda x: -x[1])

     #CI6235: Save the output image, check these image to clean data in the next step
    if rt == 1:
        out_path = data_path + dataset + "/output/"
        output_im_file_name = out_path + image.split("/")[-1].split(".")[0] + ".jpg"
        cv2.imwrite(output_im_file_name, op_im)
        cv2.imshow("output", op_im)
        cv2.waitKey(10)
        total_keep_file += 1 #CI6235: Count number of image files remaining

    free_image(im)
    free_detections(dets, num)
    return rt, idx  # Return whether having predicted vehicles and number of prediction


if __name__ == "__main__":

    net = load_net(
        "cfg/yolov3.cfg", "yolov3.weights", 0
    )  #CI6235: Load darknet model and pretrained weights
    meta = load_meta("cfg/coco.data")  # ? Load coco mdeta data
    names = ["bus", "car", "truck"]  #CI6235: Name of prediction object
    num_video = 5
    start_video = 0

     #CI6235: Choose the input video name
    for choose_data in range(start_video, num_video):
        if choose_data == 0:
            dataset = "IMG_1975"
        elif choose_data == 1:
            dataset = "IMG_1976"
        elif choose_data == 2:
            dataset = "IMG_1977"
        elif choose_data == 3:
            dataset = "IMG_1978"
        elif choose_data == 4:
            dataset = "IMG_1979"
        counts = [0, 0, 0]

        #CI6235: Main path of dataset folder
        DataDir = (
            "/CI6235/assignment1/final/data_preparation/dataset/RLR/"
            + dataset
            + "/"
        )
        
        #CI6235: Create folder to save output images
        if not os.path.exists(data_path + dataset + "/output/"):
            os.makedirs(data_path + dataset + "/output/")
            print("Directory ", data_path + dataset + "/output/", " Created ")
        else:
            print("Directory ", data_path + dataset + "/output/", " already exists")
            print("building ensemble model")
        if not os.path.exists(data_path + dataset + "/remove/"):
            os.makedirs(data_path + dataset + "/remove/")
            print("Directory ", data_path + dataset + "/remove/", " Created ")
        else:
            print("Directory ", data_path + dataset + "/remove/", " already exists")
            print("building ensemble model")

        for subdir, dirs, files in os.walk(
            DataDir
        ):  # Read frames from dataset directory
            i = 0
            files.sort(
                key=lambda f: int(filter(str.isdigit, f))
            )  # Sort files in lists in numerical order
            for file in files:
                imgfile = os.path.join(subdir, file)

                if file.endswith(".jpg"):
                    i += 1

                    #CI6235: Predict vehicles on input image
                    rt, idx = detect(
                        net, meta, imgfile, classes=1
                    )  # config file  and pretrained weights, meta data, img, model, and number of classes
                    if rt == 0:  #CI6235: Remove image file if it doen't contain vehicles
                        print("\r", "moving ", file)
                        os.rename(
                            imgfile, data_path + dataset + "/remove/" + file,
                        )
                        if os.path.isfile(subdir + file.split(".")[0] + ".txt"):
                            os.remove(subdir + file.split(".")[0] + ".txt")
                        total_remove_file += 1

                    for i in range(len(counts)):
                        counts[i] += idx[i]

        print(counts)

        #CI6235: Save the information about this step in txt file
        f = open(data_path + dataset + "/aaa_summary_auto_label.txt", "a")
        f.write("dataset, total_keep_file, total_remove_file, bus, car, truck \n")
        f.write(
            dataset
            + ","
            + str(total_keep_file)
            + ","
            + str(total_remove_file)
            + ","
            + str(counts[0])
            + ","
            + str(counts[1])
            + ","
            + str(counts[2])
        )
        f.close()

