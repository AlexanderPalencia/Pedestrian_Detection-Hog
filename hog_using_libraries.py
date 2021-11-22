#importing required libraries
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
import cv2

def calculate_hog_skimage_L2(pathImg):
    #reading the image
    img = imread(pathImg)

    #resizing image 
    # resized_img = resize(img, (128,64)) 
    # imshow(resized_img) 
    # print(resized_img.shape)
    
    hog_desc = hog(img, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), block_norm='L2', multichannel=True)

    return hog_desc

def calculate_hog_skimage_L2_from_matrix(ImgMatrix):

    hog_desc = hog(ImgMatrix, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), block_norm='L2', multichannel=True)

    return hog_desc
 
def calculate_hog_skimage_L2Hys(pathImg):
    #reading the image
    img = imread(pathImg)

    #resizing image 
    # resized_img = resize(img, (128,64)) 
    # imshow(resized_img) 
    # print(resized_img.shape)
    
    hog_desc = hog(img, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), block_norm='L2-Hys', multichannel=True)

    return hog_desc


def calculate_hog_skimage_L2Hys_from_matrix(ImgMatrix):
    
    hog_desc = hog(ImgMatrix, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), block_norm='L2-Hys', multichannel=True)

    return hog_desc

def calculate_hog_openCV(imgPath):
    img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
    # Change BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hog = cv2.HOGDescriptor()
    h = hog.compute(img)
    return h