import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import hog
import pickle
import sys
from skimage.transform import pyramid_gaussian
from skimage.util.shape import view_as_windows
import hog_using_libraries
from os.path import exists
import myNMS

# https://www.pyimagesearch.com/2017/08/28/fast-optimized-for-pixel-loops-with-opencv-and-python/

def load_pickle_model(filenameModel):
    # some time later...
    # load the model from disk
    loaded_model = pickle.load(open(filenameModel, 'rb'))
    return loaded_model

def get_subimages_of_image(imgPath, sizeSubImage, steps):
    img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
    # Change BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    matrixShape = img.shape

    imgPadding = np.pad(img, ((0,sizeSubImage[0]),(0,sizeSubImage[1]), (0,0)), 'constant')

    # img[y: y+oneSizeSaquare, x: x+oneSizeSaquare]

    subImgsList = []
    topLeftCornerList = []
    botRightCornerList = []
    for y in range(0,matrixShape[0], steps[0]):
        for x in range(0,matrixShape[1], steps[1]):
            subImg = imgPadding[y: y+sizeSubImage[0], x: x+sizeSubImage[1]]
            subImgResize = cv2.resize(subImg, (64,128))
            cv2.imwrite('./classify_sub_imgs/sub_img_{}_{}.png'.format(y,x), subImgResize)
            subImgsList.append(subImgResize)
            topLeftCornerList.append((x,y))
            botRightCornerList.append((x+sizeSubImage[1], y+sizeSubImage[0]))
    return subImgsList, topLeftCornerList, botRightCornerList, imgPadding

def predict_sub_images_from_img(subImgsListPar, topLeftCornerListPar, botRightCornerListPar, imgPaddingPar, modelPickle):
    
    predictLabel = []
    # dataframeRow = []
    for index, img in enumerate(subImgsListPar):
        f, hogToPredict = hog.calculate_HOG_from_matrix(img)
        predBin = modelPickle.predict(hogToPredict.reshape(1, -1))[0]
        predictLabel.append(predBin)
        # dataframeRow.append([subImgsListPar[index], predBin, topLeftCornerListPar[index], botRightCornerListPar[index], imgPaddingPar])
    
    # df = pd.DataFrame(dataframeRow, columns=['sub_image', 'predictLabel', 'top_left_corner','bot_rigth_corner', 'actual_image_pad'])

    return predictLabel

def draw_square_pedestria(preditionLabel, subImgsListPar, topLeftCornerListPar, botRightCornerListPar, imgPaddingPar, imgPath):
    
    img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
    # Change BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for index, prediction in enumerate(preditionLabel):
        if prediction == 1:
            # quiere decir que hay un pedestrian se dibuja el cuadro
            print('hay un pedestrian')
            start_point = topLeftCornerListPar[index]
            end_point = botRightCornerListPar[index]
            print('start point square ', start_point, ' end poing ', end_point)
            color = (0, 255, 0)
            thickness = 8
        
            detectedPedestrianImg = cv2.rectangle(img, start_point, end_point, color, thickness)

    return detectedPedestrianImg

def predict_one_img(imgPath):
    img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (64,128))
    model = load_pickle_model('./persisted_data/svm_skimage_l2.sav')
    f, hogToPredict = hog.calculate_HOG_from_matrix_optimized(img)
    # The predict function of sklearn only give us 0 or 1 but how it dicide if is 1 o 0, if is gratter than 0.5 then is 1???
    predBin = model.predict(hogToPredict.reshape(1, -1))[0]
    proba = model.predict_proba(hogToPredict.reshape(1, -1))
    print('Prediction label ', proba)

def persist_dataframe(imgNpData, featuresHOGData, labelsData, topLeftCorner, bottomRightCorner, corners, PathName):
    dfEmpty = pd.DataFrame([], columns=['Numpy_subimg_128x64', 'HOG_Features', 'predicted_label', 'top_left_corner', 'bottom_right_corner', 'position_corner_xy1_xy2'])
    dfEmpty['Numpy_subimg_128x64'] = imgNpData
    dfEmpty['HOG_Features'] = featuresHOGData
    dfEmpty['predicted_label'] = labelsData
    dfEmpty['top_left_corner'] = topLeftCorner
    dfEmpty['bottom_right_corner'] = bottomRightCorner
    dfEmpty['position_corner_xy1_xy2'] = corners
    dfEmpty.to_pickle(PathName)
    return dfEmpty

def persist_dataframe_pyramid(imgNpData, featuresHOGData, labelsData, topLeftCorner, bottomRightCorner, corners, PathName, index):
    dfEmpty = pd.DataFrame([], columns=['Numpy_subimg_128x64', 'HOG_Features', 'predicted_label', 'top_left_corner', 'bottom_right_corner', 'position_corner_xy1_xy2'])
    dfEmpty['Numpy_subimg_128x64'] = imgNpData
    dfEmpty['HOG_Features'] = featuresHOGData
    dfEmpty['predicted_label'] = labelsData
    dfEmpty['top_left_corner'] = topLeftCorner
    dfEmpty['bottom_right_corner'] = bottomRightCorner
    dfEmpty['position_corner_xy1_xy2'] = corners
    dfEmpty.to_pickle(PathName)
    return dfEmpty

def test__decision_function_model(imgPath, modelPath):
    print('entrando')
    model = load_pickle_model(modelPath)
    # Read image
    img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
    # Change BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = cv2.resize(img, (64,128))

    hogToPredict = hog_using_libraries.calculate_hog_skimage_L2_from_matrix(img)
    predictDecisionFunct = model.decision_function(hogToPredict.reshape(1, -1))

    print('Holaaaa, ', predictDecisionFunct)

def pipeline_detect_pedestrian_pyramid_method(imgPath, modelPath, imgName):
    # Create folder to save results
    if (not exists('./results/{}'.format(imgName))):
        os.mkdir('./results/{}'.format(imgName))
        print('folder {} created successfully'.format(imgName))
    model = load_pickle_model(modelPath)

    # Read image
    img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
    # Change BGR to RGB
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Making copies of the images so we can make detection squares
    imgOriginalSize = np.copy(img)
    imgOriginalSize1 = np.copy(img)
    imgOriginalSize3 = np.copy(img)
    
    # Windows shape size NxMx3
    window_shape = (128, 64, 3)
    # Steps that the subimage is going to move in x and y
    window_step = 32

    # declarations of list variables so we can persist or dataframe
    subImgListPedestrian = []
    hogListPedestrian = []
    topLeftCornerList = []
    botRightCornerList = []
    labelPredictList = []
    coordinatesXY1XY2List = []
    rescaleCoordinatesXY1XY2List = []
    rescaleCordenatesTopLeftCorner = []
    rescaleCordenatesbotrightCorner = []
    probabilityPredictionList = []

    # Factor that we apply in the pyramid method (risize by this factor)
    down_scale_factor = 1.2

    # img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
    pyramid = tuple(pyramid_gaussian(img, downscale=down_scale_factor))

    # Iterate over images generated by the pyramid method
    for index,p in enumerate(pyramid):
        # normalize image
        imgPyramid = cv2.normalize(p, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        imgPyramid = cv2.cvtColor(imgPyramid, cv2.COLOR_BGR2RGB)
        
        # Checking that the size of the subimage is gratter than the windows shapes.
        if (imgPyramid.shape[:-1] < window_shape[:-1]):
            print('The size of the pyramid image is smaller than that of the windows size')
            break

        # Creating copies of the pyramid image to draw predictions squares
        imgNoNMS = np.copy(imgPyramid)
        # imgYesNMS = np.copy(imgPyramid)

        # Creating sub-images or bounding boxes of the image to predict
        windows = view_as_windows(imgPyramid, window_shape, step=window_step)
        print('Image size ',imgPyramid.shape)
        print('Windows sizes ',windows.shape)

        # iterete over the bounding boxes
        # a is the y in the image
        for a in range(windows.shape[0]):
            # b is the x in the image
            for b in range(windows.shape[1]):
                # Slicing over the matrix to get one bounding box to predict
                i = windows[a,b,0,:,:,:]

                # Passing bounding box image to hog
                hogToPredict = hog_using_libraries.calculate_hog_skimage_L2_from_matrix(i)

                # Predicting over the hog features
                predBin = model.predict(hogToPredict.reshape(1, -1))[0]

                # Get prediction between [-1,1]
                predictDecisionFunct = model.decision_function(hogToPredict.reshape(1, -1))[0]

                # Threshold of the decision function if is gratter than x it means that it is possitive.
                if predictDecisionFunct >= 0.30:
                    print('Decision function predict: ', predictDecisionFunct)

                    # Find a pedestrian persist data
                    subImgListPedestrian.append(i)
                    hogListPedestrian.append(hogToPredict)
                    labelPredictList.append(predBin)
                    probabilityPredictionList.append(predictDecisionFunct)

                    # x, y top left corner coordinates
                    topLeftCornerList.append((b*window_step, a*window_step))
                    # x, y bottom right corner coordinates
                    botRightCornerList.append(((b*window_step)+window_shape[1], (a*window_step)+window_shape[0]))

                    # Rescaling coordinates
                    newRescaleCoordinatesLeft = (int((b*window_step) * (down_scale_factor**index)), int((a*window_step) * (down_scale_factor**index)))

                    newRescaleCoordinatesRight = (int(((b*window_step)+window_shape[1]) * (down_scale_factor**index)), int(((a*window_step)+window_shape[0]) * (down_scale_factor**index)))

                    # Persist RESCALE CORNERS FOR ORIGINAL IMAGE
                    rescaleCordenatesTopLeftCorner.append(newRescaleCoordinatesLeft)
                    rescaleCordenatesbotrightCorner.append(newRescaleCoordinatesRight)

                    rescaleCoordinatesSquareList = list(newRescaleCoordinatesLeft+newRescaleCoordinatesRight)
                    rescaleCoordinatesXY1XY2List.append(rescaleCoordinatesSquareList)

                    # Create a list of coordinates to pass NMS
                    coordinatesSquareList = [b*window_step, a*window_step, (b*window_step)+window_shape[1], (a*window_step)+window_shape[0]]
                    coordinatesXY1XY2List.append(coordinatesSquareList)

                    # defing atributes
                    color = (0, 255, 0)
                    thickness = 8

                    # Drawing square prediction in image
                    cv2.rectangle(imgNoNMS, (b*window_step, a*window_step), ((b*window_step)+window_shape[1], (a*window_step)+window_shape[0]), color, thickness)

        # Saving all pyramid image with they squares.
        imgNewPath = './results/{}/predicted_pyramid_no_NMS_{}_{}.png'.format(imgName, index,imgName)
        correctColorSubImages = cv2.cvtColor(imgNoNMS, cv2.COLOR_RGB2BGR)
        cv2.imwrite(imgNewPath, correctColorSubImages)

    # Rescaling the coordinates to the original image
    print('rescale cooridinates')
    for leftTop, rigthBot, proba in zip(rescaleCordenatesTopLeftCorner, rescaleCordenatesbotrightCorner, probabilityPredictionList):
        # Creating and definig the attributes
        color1 = (0, 255, 0)
        thickness = 4
        # Drawing squares
        detectedPedestrianImgOriginalSizeNoNMS = cv2.rectangle(imgOriginalSize, leftTop, rigthBot, color1, thickness)
        intProba = round(proba*100)
        detectedPedestrianImgOriginalSizeNoNMS = cv2.putText(imgOriginalSize, str(intProba), leftTop, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 3)
        

    # Saving the original image with the squares predicted in pyramid with correct coordinats
    imgNewPath = './results/{}/predicted_pyramid_NO_NMS_Original_img_{}_{}.png'.format(imgName, index,imgName)
    img2 = cv2.cvtColor(detectedPedestrianImgOriginalSizeNoNMS, cv2.COLOR_RGB2BGR)
    cv2.imwrite(imgNewPath, detectedPedestrianImgOriginalSizeNoNMS)

    # Start NMS Algorithm
    print('NMS Algorithm start....')
    print('Total squares predictions ', len(rescaleCoordinatesXY1XY2List))
    # Converting list of coordinates in np array
    arraycoordinates = np.array(rescaleCoordinatesXY1XY2List)
    # Pass coordinates to NMS Algorith to reduce squares
    newCoordinates, newProbaList = myNMS.NMS_score(arraycoordinates, probabilityPredictionList, 0.8)
    print('Total squares NMS ', len(newCoordinates))
    # Iterate over new coordinates to draw saqueres
    for coordinate, proba in zip(newCoordinates, newProbaList):
        topNewLeftCorner = tuple(coordinate[0:2])
        rigthNewBoCorner = tuple(coordinate[2:])
        color = (0, 255, 0)
        thickness = 8
        # draw saqueres in the original images
        detectedPedestrianImgYesNMS = cv2.rectangle(imgOriginalSize1, topNewLeftCorner, rigthNewBoCorner, color, thickness)
        intProba = round(proba*100)
        detectedPedestrianImgYesNMS = cv2.putText(imgOriginalSize1, str(intProba), topNewLeftCorner, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 5)

        
    # Save images
    imgNewPath = './results/{}/predicted_yes_NMS_SCORE_{}.png'.format(imgName,imgName)
    img7 = cv2.cvtColor(detectedPedestrianImgYesNMS, cv2.COLOR_RGB2BGR)
    cv2.imwrite(imgNewPath, detectedPedestrianImgYesNMS)



    newCoordinatesNoScore = myNMS.NMS_no_score(arraycoordinates, 0.8)
    # Iterate over new coordinates to draw saqueres
    for coordinate in newCoordinatesNoScore:
        topNewLeftCorner = tuple(coordinate[0:2])
        rigthNewBoCorner = tuple(coordinate[2:])
        color = (0, 255, 0)
        thickness = 8
        # draw saqueres in the original images
        detectedPedestrianImgYesNMSNoScore = cv2.rectangle(imgOriginalSize3, topNewLeftCorner, rigthNewBoCorner, color, thickness)

    
    # Save images
    imgNewPath = './results/{}/predicted_yes_NMS_NO_SCORE_{}.png'.format(imgName,imgName)
    img7 = cv2.cvtColor(detectedPedestrianImgYesNMSNoScore, cv2.COLOR_RGB2BGR)
    cv2.imwrite(imgNewPath, detectedPedestrianImgYesNMSNoScore)
    
    


    # Persist variables to dataset
    persist_dataframe(subImgListPedestrian, hogListPedestrian, labelPredictList, topLeftCornerList, botRightCornerList, coordinatesXY1XY2List, './results/{}/df_pyramid_{}.pkl'.format(imgName,imgName))

    # return detectedPedestrianImgNoNMS

def pipeline_detect_pedestrian_normal_method(imgPath, modelPath, imgName):
    print('Detecting pedestrian in {} image'.format(imgName))
    if (not exists('./results/{}'.format(imgName))):
        os.mkdir('./results/{}'.format(imgName))
        print('folder {} created successfully'.format(imgName))
    model = load_pickle_model(modelPath)
    # Read image
    img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
    # Change BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    imgNoNMS = np.copy(img)
    imgYesNMS = np.copy(img)
    
    print('Image size ',img.shape)
    subImgListPedestrian = []
    hogListPedestrian = []
    topLeftCornerList = []
    botRightCornerList = []
    coordinatesXY1XY2List = []
    labelPredictList = []

    # windowsShapesList = [(400, 200, 3), (500, 250, 3)]
    windowsShapesList = [(500, 250, 3)]
    # window_shape = (500, 250, 3)
    window_step = 32
    for index, window_shape in enumerate(windowsShapesList):
        windows = view_as_windows(img, window_shape, step=window_step)
        print('Windows sizes ',windows.shape)

        for a in range(windows.shape[0]):
            # a is the y in the image
            for b in range(windows.shape[1]):
                # b is the x in the image
                i = windows[a,b,0,:,:,:]
                subImgResize = cv2.resize(i, (64,128))

                # hogToPredict = hog_using_libraries.calculate_hog_skimage_L2Hys_from_matrix(subImgResize)
                
                f, hogToPredict = hog.calculate_HOG_from_matrix_optimized(subImgResize)
                predBin = model.predict(hogToPredict.reshape(1, -1))[0]
                if predBin == 1:
                    # Find a pedestrian persist data
                    subImgListPedestrian.append(i)
                    hogListPedestrian.append(hogToPredict)
                    labelPredictList.append(predBin)
                    # coordinates are x, y
                    topLeftCornerList.append((b*window_step, a*window_step))
                    
                    # coordinates are x, y
                    botRightCornerList.append(((b*window_step)+window_shape[1], (a*window_step)+window_shape[0]))

                    # Create a list of coordinates to pass NMS
                    coordinatesSquareList = [b*window_step, a*window_step, (b*window_step)+window_shape[1], (a*window_step)+window_shape[0]]
                    coordinatesXY1XY2List.append(coordinatesSquareList)

                    color = (0, 255, 0)
                    thickness = 8
                
                    detectedPedestrianImgNoNMS = cv2.rectangle(imgNoNMS, (b*window_step, a*window_step), ((b*window_step)+window_shape[1], (a*window_step)+window_shape[0]), color, thickness)

    persist_dataframe(subImgListPedestrian, hogListPedestrian, labelPredictList, topLeftCornerList, botRightCornerList, coordinatesXY1XY2List, './results/{}/df_{}.pkl'.format(imgName,imgName))
    

    print('NMS Algorithm start....')
    print('Total squares predictions ', len(coordinatesXY1XY2List))
    arraycoordinates = np.array(coordinatesXY1XY2List)
    newCoordinates = myNMS.NMS(arraycoordinates, 0.4)
    print('Total squares NMS ', len(newCoordinates))
    for coordinate in newCoordinates:
        topNewLeftCorner = tuple(coordinate[0:2])
        rigthNewBoCorner = tuple(coordinate[2:])
        color = (0, 255, 0)
        thickness = 8
    
        detectedPedestrianImgYesNMS = cv2.rectangle(imgYesNMS, topNewLeftCorner, rigthNewBoCorner, color, thickness)


    plt.imshow(detectedPedestrianImgNoNMS)
    plt.title('Predict Pedestrians NO NMS')
    imgNewPath = './results/{}/predicted_no_NMS_{}.png'.format(imgName,imgName)

    
    img1 = cv2.cvtColor(detectedPedestrianImgNoNMS, cv2.COLOR_RGB2BGR)
    cv2.imwrite(imgNewPath, img1)
    plt.show()

    
    
    plt.imshow(detectedPedestrianImgYesNMS)
    plt.title('Predict Pedestrians YES NMS')
    imgNewPath = './results/{}/predicted_yes_NMS_{}.png'.format(imgName,imgName)
    
    img2 = cv2.cvtColor(detectedPedestrianImgYesNMS, cv2.COLOR_RGB2BGR)
    cv2.imwrite(imgNewPath, img2)
    plt.show()


    imgNewPath = './results/{}/original_image{}.png'.format(imgName,imgName)
    img3 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(imgNewPath, img3)
    
    return detectedPedestrianImgNoNMS

def check_corners_windows():
    A = np.arange(5*4).reshape(5, 4)
    print(A)
    window_shape = (2, 2)
    window_step = 1

    B = view_as_windows(A, window_shape, window_step)
    print(B.shape)
    
    for a in range(B.shape[0]):
        # a is the y in the image
        for b in range(B.shape[1]):
            # b is the x in the image
            i = B[a,b,:,:]
            print(b*window_step, a*window_step, ' top right corner ', (b*window_step)+window_shape[1], (a*window_step)+window_shape[0])
            print(A[a*window_step: (a*window_step)+window_shape[0] ,b*window_step: (b*window_step)+window_shape[1]])

if __name__ == "__main__":
    # predict_one_img('./dataset/test_single_subimages/test3.png')
    # a = pipeline_detect_pedestrian_pyramid_method('./dataset/test_eval/052.jpg', './persisted_data/svm_my_hog.sav', '052')



    b = pipeline_detect_pedestrian_pyramid_method('./dataset/test_eval/test_train.jpg', './persisted_data/svm_my_hog.sav', 'test_train')

    # if len(sys.argv) == 1:
    #     print('Faltan parametros para correcto funcionamiento path de imagen y nombre imagen')
    # else:
    #     img_src = str(sys.argv[1])
    #     imgName = str(sys.argv[2])
    #     pipeline_detect_pedestrian_pyramid_method(img_src, './persisted_data/svm_skimage_l2.sav', 'Modelo_sklearn_'+imgName)
    #     pipeline_detect_pedestrian_pyramid_method(img_src, './persisted_data/svm_my_hog.sav', 'Modelo_My_sklearn_'+imgName)

    
    # a = pipeline_detect_pedestrian_normal_method('./dataset/test_eval/test_train.jpg', './persisted_data/svm_my_hog.sav', 'test_train')




    # test__decision_function_model('./dataset/test_single_subimages/test4.png', './persisted_data/svm_my_hog.sav')




# project 1\dataset\test_single_subimages\test1.png



    # if len(sys.argv) == 1:
    #     print('Faltan o sobran parametros para correcto funcionamiento')
    # else:
    #     img_src_path = str(sys.argv[1])
    #     print('Buscando peatones en la imagen ', img_src_path)

    #     model = load_pickle_model('smv_model.sav')
    #     print('Modelo cargado correctamente ')

    #     listSize = [(128,64), (256,128), (400,200), (600,300)]
    #     listSizeMove = [(128,64), (256,128, (400,200), (64,32))]

    #     totalsubImgsList = []
    #     totaltopLeftCornerList = []
    #     totalbotRightCornerList = []
    #     totalimgPadding = []

    #     for i in listSize:
    #         for a in listSizeMove:
    #             subImgsList, topLeftCornerList, botRightCornerList, imgPadding = get_subimages_of_image(img_src_path, i, a)
    #             totalsubImgsList += subImgsList
    #             totaltopLeftCornerList += topLeftCornerList
    #             totalbotRightCornerList += botRightCornerList
    #             totalimgPadding = imgPadding

    #     print('Generando sub imagenes ', len(totalsubImgsList))
    #     plabels = predict_sub_images_from_img(totalsubImgsList, totaltopLeftCornerList, totalbotRightCornerList, totalimgPadding, model)
        
    #     print('Prediciendo sub imagenes ....')

    #     a = draw_square_pedestria(plabels, totalsubImgsList, totaltopLeftCornerList, totalbotRightCornerList, totalimgPadding, img_src_path)
            
    #     plt.imshow(a)
    #     plt.title('Predict Pedestrians')
    #     imgNewPath = './resultado_{}'.format(img_src_path)
    #     print('imagen almacenaada en ',imgNewPath)
    #     cv2.imwrite(imgNewPath, a)
            
    #     # plt.imsave()
        
    #     plt.show()


