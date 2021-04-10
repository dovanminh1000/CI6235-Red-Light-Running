#CI6235: Reference: https://keras.io/api/preprocessing/image/#flow-method
#CI635: import the necessary packages

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import argparse
import os


# load the input image, convert it to a NumPy array, and then
# reshape it to have an extra dimension

data_path = '/CI6235/assignment1/final/data_preparation/dataset/RLR/'

f = open(data_path + "aaa_summary_data_aug.txt", "w")
f.close()

total_frame = 0
total_frame_each_video = 0
num_video = 4
num_max_frame = 2

#CI6235: Choose input image
for choose_video in range(0, num_video):
	if choose_video == 0:
		img_prefix = "IMG_1975"
	elif choose_video == 1:
		img_prefix = "IMG_1976"
	elif choose_video == 2:
		img_prefix = "IMG_1977"
	elif choose_video == 3:
		img_prefix = "IMG_1978"
	elif choose_video == 4:
		img_prefix = "IMG_1979"

	try:
		os.makedirs(os.path.join(data_path, img_prefix ))
	except FileExistsError:
		print(
					"Error: Folder already exists - "
						+ os.path.join(data_path, img_prefix)
				)

	arr = os.listdir(data_path+img_prefix)
	# get video name by using split method

	print("[INFO] loading example image...")
	for num in range(len(arr)):
		video_name = arr[num].split('.')[0]
		image = load_img(data_path + img_prefix+"/"+ arr[num])
		image = img_to_array(image)
		image = np.expand_dims(image, axis=0)

		#CI6235: image generator for data augmentation then
		aug = ImageDataGenerator(
			rotation_range=20,
			zoom_range=0.15,
			shear_range=0.15,
			horizontal_flip=False,
			fill_mode="nearest")
		total = 0  #CI6235: initialize the total number of images generated 
		# construct the actual Python generator
		print("Generating images...: ", img_prefix, num)
		#CI6235: Generate new images
		imageGen = aug.flow(image, batch_size=1, save_to_dir=data_path+img_prefix,
			save_prefix=video_name, save_format="jpg") 

		# loop over examples from our image data augmentation generator
		for image in imageGen:
			# increment our counter
			total += 1
			total_frame+=1
			total_frame_each_video+=1

			# if we have reached the specified number of examples, break
			# from the loop
			if total == num_max_frame:
				break
	#CI6235: Summarise data generated
	f = open(data_path + "aaa_summary_data_aug.txt", "a")
	f.write(img_prefix + ": " + str(total_frame_each_video) + "frames \n")
	f.close()

f = open(data_path + "aaa_summary_data_aug.txt", "a")
f.write("Total frames: " + str(total_frame) + "frames \n")
f.close()
