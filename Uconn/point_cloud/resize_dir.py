import cv2
import os
import glob

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def resize_dir(dir_string, new_image_width):
    images = sorted(glob.glob(dir_string + '*.jpg'))
    folders = glob.glob(dir_string + '*/')
    
    if not (dir_string + 'RESIZED/') in folders:
        os.mkdir(dir_string + 'RESIZED')
        print('Created RESIZED directory')

    for image_name in images:
        name_no_parent_folder = (os.path.split(image_name)[1]) #drop parent folder
        image = cv2.imread(image_name)
        resized = image_resize(image, width=new_image_width)
        cv2.imwrite(dir_string + 'RESIZED/' + name_no_parent_folder, resized)

resize_width = 600

resize_dir('./images/stereoRight/', resize_width)
resize_dir('./images/stereoRight/', resize_width)