import numpy as np
import cv2
from matplotlib import pyplot as plt
import math 
import os

def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()

def apply_threshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)

    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)

    return matrix

def simplest_cb(img, percent):
    assert img.shape[2] == 3
    assert percent > 0 and percent < 100

    half_percent = percent / 200.0

    channels = cv2.split(img)

    out_channels = []
    for channel in channels:
        assert len(channel.shape) == 2
        # find the low and high precentile values (based on the input percentile)
        height, width = channel.shape
        vec_size = width * height
        flat = channel.reshape(vec_size)

        assert len(flat.shape) == 1

        flat = np.sort(flat)

        n_cols = flat.shape[0]

        low_val  = flat[math.floor(n_cols * half_percent)]
        high_val = flat[math.ceil( n_cols * (1.0 - half_percent))]

        #print("Lowval: ", low_val)
        #print("Highval: ", high_val)

        # saturate below the low percentile and above the high percentile
        thresholded = apply_threshold(channel, low_val, high_val)
        # scale the channel
        normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized)

    return cv2.merge(out_channels)

#Function that Downsamples image x number (reduce_factor) of times. 
def downsample_image(image, reduce_factor):
	for i in range(0,reduce_factor):
		#Check if image is color or grayscale
		if len(image.shape) > 2:
			row,col = image.shape[:2]
		else:
			row,col = image.shape

		image = cv2.pyrDown(image, dstsize= (col//2, row // 2))
	return image

#Function to create point cloud file
def create_output(vertices, colors, filename):
	colors = colors.reshape(-1,3)
	vertices = np.hstack([vertices.reshape(-1,3),colors])

	ply_header = '''ply
		format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		property uchar red
		property uchar green
		property uchar blue
		end_header
		'''
	with open(filename, 'w') as f:
		f.write(ply_header %dict(vert_num=len(vertices)))
		np.savetxt(f,vertices,'%f %f %f %d %d %d')

def generate_ply(image):

    # Load the DNN model
    model = cv2.dnn.readNet("models/model-f6b98070.onnx")

    if (model.empty()):
        print("Could not load the neural net! - Check path")

    # Set backend and target to CUDA to use GPU
    #model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    #model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    cv_file = cv2.FileStorage()
    cv_file.open('StereoMaps/stereoMap.xml', cv2.FileStorage_READ)

    Q = cv_file.getNode('q').mat()
    
    img = cv2.imread(image)
    img = simplest_cb(img, 50)

    cv2.imshow('image', img)
    cv2.waitKey(0)

    img = downsample_image(img, 2)

    cv2.imshow('image', img)
    cv2.waitKey(0)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    imgHeight, imgWidth, channels = img.shape

    # Create Blob from Input Image
    # MiDaS v2.1 Large ( Scale : 1 / 255, Size : 384 x 384, Mean Subtraction : ( 123.675, 116.28, 103.53 ), Channels Order : RGB )
    blob = cv2.dnn.blobFromImage(img, 1/255., (384,384), (123.675, 116.28, 103.53), True, False)

    # Set input to the model
    model.setInput(blob)

    # Make forward pass in model
    output = model.forward()

    output = output[0,:,:]
    output = cv2.resize(output, (imgWidth, imgHeight))

    # Normalize the output
    output = cv2.normalize(output, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    plt.imshow(output, 'gray')
    plt.show()

    # -------------------------------------------------------------------------------------

    #Reproject points into 3D
    points_3D = cv2.reprojectImageTo3D(output, Q, handleMissingValues=False)

    #Get rid of points with value 0 (i.e no depth)
    mask_map = output > output.min()

    #Mask colors and points. 
    output_points = points_3D[mask_map]
    output_colors = img[mask_map]
    
    name_no_extention = (os.path.splitext(image)[0]) #drop extention
    name_no_folder = (os.path.split(name_no_extention)[1]) #drop parent folder

    output_file = './plys/' + name_no_folder + '.ply'
    #Generate point cloud 
    create_output(output_points, output_colors, output_file)

    cv2.destroyAllWindows()

generate_ply('./images/37000/37000 - E1 - 2021-04-14 - 14-48.jpg')
generate_ply('./images/37000/37000 - H1 - 2021-04-14 - 14-48.jpg')