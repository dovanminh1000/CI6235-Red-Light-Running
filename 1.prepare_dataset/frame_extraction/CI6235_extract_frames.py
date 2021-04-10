#CI6235: Generate frame from input video
import cv2
import os

#CI6235: Input and output data
img_prefix = "RLR"
input_loc = "/dataset/CI6235/RLR/"
output_loc = "../dataset/" + img_prefix + "/"

skipFrames = 30 #CI6235: Number of frames skipping
resizeFactor = 3 #CI6235: Resize frame factor

video_name = ""
num_video = 5
total_frame = 0
total_frame_each_video = 0

#CI6235: Create folder to save output data
try:
    os.makedirs(os.path.join(output_loc, video_name.split(".")[0]))
except FileExistsError:
    print(
                "Error: Folder already exists - "
                + os.path.join(output_loc, video_name.split(".")[0])
            )
f = open(output_loc + "aaa_summary.txt", "w")
f.close()

for choose_video in range(0, num_video):
    if choose_video == 0:
        img_prefix = "IMG_1975"
        video_name = "IMG_1975.MOV"

    elif choose_video == 1:
        img_prefix = "IMG_1976"
        video_name = "IMG_1976.MOV"

    elif choose_video == 2:
        img_prefix = "IMG_1977"
        video_name = "IMG_1977.MOV"
    elif choose_video == 3:
        img_prefix = "IMG_1978"
        video_name = "IMG_1978.MOV"
    
    elif choose_video == 4:
        img_prefix = "RLR_1979"
        video_name = "IMG_1979.MOV"
    total_frame_each_video = 0
    if True:
        print("Processing Vid ", video_name)
        print(os.path.join(input_loc, video_name))
        cap = cv2.VideoCapture(os.path.join(input_loc, video_name))
        frame_counter = 0

	#CI6235: Create folder to save output data
        try:
            os.makedirs(os.path.join(output_loc, video_name.split(".")[0]))
        except FileExistsError:
            print(
                        "Error: Folder already exists - "
                        + os.path.join(output_loc, video_name.split(".")[0])
                    )
            # continue
        frame_counter = 0
        while cap.isOpened():
            for i in range(skipFrames): #CI6235: for loop to skip frames
                ret, frame = cap.read()
            if ret == True:
                frame = cv2.resize(
                    frame,
                    (frame.shape[1] // resizeFactor, frame.shape[0] // resizeFactor),
                ) #CI6235: Resize original frame to small size
                output_path = os.path.join(
                    output_loc,
                    video_name.split(".")[0],
                    img_prefix + "_" + str(frame_counter) + ".jpg",
                ) #CI6235: Save the output frame
                total_frame_each_video+=1
                total_frame+=1
                cv2.imwrite(output_path, frame)
                print(output_path)
            else:
                print("done")
                break
            frame_counter += 1
	#CI6235: Summarise information from this step
        f = open(output_loc + "aaa_summary.txt", "a")
        f.write(img_prefix + ": " + str(total_frame_each_video) + "frames \n")
        f.close()
        cap.release()
        cv2.destroyAllWindows()
f = open(output_loc + "aaa_summary.txt", "a")
f.write("total frames: " + str(total_frame) + "frames \n")
f.close()

