import pandas as pd
import os
import cv2
import hog
import hog_using_libraries
import time

def generate_dataframe_my_hog(backgroundFolderPath, pedestrianFolderPath):
    featuresArray = []
    labels = []
    imgs = []
    count = 0
    imagesNames = os.listdir(backgroundFolderPath)
    for imageName in imagesNames:
        count += 1
        image_path = os.path.join(backgroundFolderPath, imageName)
        
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # Change BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)

        featuresSquares105, featuresOneList = hog.calculate_HOG_optimized(image_path)
        
        featuresArray.append(featuresOneList)
        labels.append(0)

    imagesNames = os.listdir(pedestrianFolderPath)
    for imageName in imagesNames:
        count += 1
        image_path = os.path.join(pedestrianFolderPath, imageName)

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # Change BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)

        featuresSquares105, featuresOneList = hog.calculate_HOG_optimized(image_path)

        featuresArray.append(featuresOneList)
        labels.append(1)
    
    print('Finish data successfully MY HOG OPTIMIZED, ', count, ' images transformed')
    return persist_dataframe(imgs, featuresArray, labels, './persisted_data/df_my_hog_function.pkl')

def generate_dataframe_skimage_l2(backgroundFolderPath, pedestrianFolderPath):
    featuresArray = []
    labels = []
    imgs = []
    count = 0
    errors = []
    imagesNames = os.listdir(backgroundFolderPath)
    for imageName in imagesNames:
        count += 1
        image_path = os.path.join(backgroundFolderPath, imageName)
        
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # Change BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)

        try:
            featuresOneList = hog_using_libraries.calculate_hog_skimage_L2(image_path)
            featuresArray.append(featuresOneList)
            labels.append(0)
        except:
            errors.append(['Error Processing img', imageName])
            featuresArray.append(['Error Processing img', imageName])
            labels.append(-1)




    imagesNames = os.listdir(pedestrianFolderPath)
    for imageName in imagesNames:
        count += 1
        image_path = os.path.join(pedestrianFolderPath, imageName)

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # Change BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)

        try:
            featuresOneList = hog_using_libraries.calculate_hog_skimage_L2(image_path)
            featuresArray.append(featuresOneList)
            labels.append(1)
        except:
            errors.append(['Error Processing img', imageName])
            featuresArray.append(['Error Processing img', imageName])
            labels.append(-1)
    
    print('Finish data successfully Skimage L2', count, ' images transformed errors ', len(errors), errors)

    return persist_dataframe(imgs, featuresArray, labels, './persisted_data/df_hog_skimage_l2.pkl')

def generate_dataframe_skimage_l2_hys(backgroundFolderPath, pedestrianFolderPath):
    """
    La imagen AnnotationsNeg_0.000000_D2004-08-19_15h56m44s_1.png al pasarla da un error esto debido a que es una imagen totalmente en blanco, skimage como se le puso multychannel true esta imagen la agaara como de una sola dimmension.
    """
    featuresArray = []
    labels = []
    imgs = []
    count = 0
    errors = []
    imagesNames = os.listdir(backgroundFolderPath)
    for imageName in imagesNames:
        count += 1
        image_path = os.path.join(backgroundFolderPath, imageName)
        
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # Change BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
        try:
            featuresOneList = hog_using_libraries.calculate_hog_skimage_L2Hys(image_path)
            featuresArray.append(featuresOneList)
            labels.append(0)
        except:
            errors.append(['Error Processing img', imageName])
            featuresArray.append(['Error Processing img', imageName])
            labels.append(-1)

    imagesNames = os.listdir(pedestrianFolderPath)
    for imageName in imagesNames:
        count += 1
        image_path = os.path.join(pedestrianFolderPath, imageName)

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # Change BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
        
        try:
            featuresOneList = hog_using_libraries.calculate_hog_skimage_L2Hys(image_path)
            featuresArray.append(featuresOneList)
            labels.append(1)
        except:
            errors.append(['Error Processing img', imageName])
            featuresArray.append(['Error Processing img', imageName])
            labels.append(-1)
    
    print('Finish data successfully Skimage L2 - HYS', count, ' images transformed errors ', len(errors), errors)

    return persist_dataframe(imgs, featuresArray, labels, './persisted_data/df_hog_skimage_l2_hys.pkl')

def generate_dataframe_opencv_l2_hys(backgroundFolderPath, pedestrianFolderPath):
    featuresArray = []
    labels = []
    imgs = []
    count = 0
    imagesNames = os.listdir(backgroundFolderPath)
    for imageName in imagesNames:
        count += 1
        image_path = os.path.join(backgroundFolderPath, imageName)
        
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # Change BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)

        featuresOneList = hog_using_libraries.calculate_hog_openCV(image_path)

        featuresArray.append(featuresOneList)
        labels.append(0)

    imagesNames = os.listdir(pedestrianFolderPath)
    for imageName in imagesNames:
        count += 1
        image_path = os.path.join(pedestrianFolderPath, imageName)

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # Change BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)

        featuresOneList = hog_using_libraries.calculate_hog_openCV(image_path)

        featuresArray.append(featuresOneList)
        labels.append(1)
    
    print('Finish data successfully OpenCV L2-HYS', count, ' images transformed')
    
    return persist_dataframe(imgs, featuresArray, labels, './persisted_data/df_hog_opencv.pkl')

def persist_dataframe(imgNpData, featuresHOGData, labelsData, PathName):
    dfEmpty = pd.DataFrame([], columns=['Numpy_img', 'HOG_Features', 'label'])
    dfEmpty['Numpy_img'] = imgNpData
    dfEmpty['HOG_Features'] = featuresHOGData
    dfEmpty['label'] = labelsData
    dfEmpty.to_pickle(PathName)
    return dfEmpty

if __name__ == "__main__":
    # backgroundFolder = r'./dataset/Pedestrians-Dataset-Dummy/Background'
    # pedestrianFolder = r'./dataset/Pedestrians-Dataset-Dummy/Pedestrians'

    backgroundFolder = r'./dataset/Pedestrians-Dataset/Background'
    pedestrianFolder = r'./dataset/Pedestrians-Dataset/Pedestrians'

    start_time = time.time()
    generate_dataframe_skimage_l2(backgroundFolder, pedestrianFolder)
    print("--- %s seconds --- Datraframe Save Skimage L2 ---" % (time.time() - start_time))
    
    start_time = time.time()
    generate_dataframe_skimage_l2_hys(backgroundFolder, pedestrianFolder)
    print("--- %s seconds --- Datraframe Save Skimage L2 HYS ---" % (time.time() - start_time))

    start_time = time.time()
    generate_dataframe_opencv_l2_hys(backgroundFolder, pedestrianFolder)
    print("--- %s seconds --- Datraframe Save OpenCV L2 HYS ---" % (time.time() - start_time))
        
    start_time = time.time()
    generate_dataframe_my_hog(backgroundFolder, pedestrianFolder)
    print("--- %s seconds --- Datraframe Save My HOG Optimized ---" % (time.time() - start_time))

