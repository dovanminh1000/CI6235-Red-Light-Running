import os
import shutil

auto_dataset = "/CI6235/assignment1/final/data_preparation/dataset/RLR/auto_label/"
manual_dataset = "/CI6235/assignment1/final/data_preparation/dataset/RLR/manual_label/"
dataset_path = "/CI6235/assignment1/final/data_preparation/dataset/RLR/"
num_video = 5
start_video = 0
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

    if not os.path.exists(manual_dataset + dataset + "/dataset/"):
        os.makedirs(manual_dataset + dataset + "/dataset/")
        print("Directory ", manual_dataset + dataset + "/dataset/", " Created ")
    else:
        print("Directory ", manual_dataset + dataset + "/dataset/", " already exists")
        print("building ensemble model")
    #CI6235: Keep the cleaned data after manually checking the data generated in auto labeling step
    for subdir, dirs, files in os.walk(
        auto_dataset + dataset + "/output"
    ):  # Read frames from dataset directory
        print("subdir: ", subdir)
        print("dirs: ", dirs)
        for file in files:
            imgfile = os.path.join(subdir, file)
            if file.endswith(".jpg"):
                label_name = file.split(".")[0] + ".txt"
                print("imgfile: ", file)
                print("imgfile: ", label_name)
                shutil.copy(
                    dataset_path + dataset + "/" + file,
                    manual_dataset + dataset + "/dataset/",
                )
                shutil.copy(
                    dataset_path + dataset + "/" + label_name,
                    manual_dataset + dataset + "/dataset/",
                )

