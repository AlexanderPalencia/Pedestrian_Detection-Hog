# Manuel Alexander Palencia Gutierrez
from numpy.core.fromnumeric import shape
import numpy as np
import cv2
import bisect
import hog_using_libraries
import math
import time

def calculate_gradient_x_y(imgPath):
    # Read image
    img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
    # Change BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # imgFloat = np.float32(img)
    imgFloat = np.float32(img/255.0)
    gx = cv2.Sobel(imgFloat, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(imgFloat, cv2.CV_32F, 0, 1, ksize=1)
    return gx, gy

def calculate_gradient_x_y_from_matrix(imgMatrix):
    # imgFloat = np.float32(img)
    imgFloat = np.float32(imgMatrix/255.0)
    gx = cv2.Sobel(imgFloat, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(imgFloat, cv2.CV_32F, 0, 1, ksize=1)
    return gx, gy

def calculate_magnitude_orientation(gradientX, gradientY):
    # mag = np.sqrt((gradientX ** 2) + (gradientY ** 2))
    mag = np.sqrt(np.add(np.square(gradientX), np.square(gradientY)))
    orientation = np.arctan2(gradientY, gradientX) * (180 / np.pi) % 180
    # orientation = np.arctan2(gradientX, gradientY)
    return mag, orientation

def calculate_magnitude_orientation_opencv(gx,gy):
    # Python Calculate gradient magnitude and direction ( in degrees )
    mag, orientation = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    unsiendOri = orientation % 180
    return mag, unsiendOri

def calculate_max_magnitude_orientation(magMatrix, orientationMAtrix):
    channelRed = magMatrix[:,:,0]
    channelGreen = magMatrix[:,:,1]
    channelBlue = magMatrix[:,:,2]
    
    maxMagValArray = []
    maxIndexChannelArray = []
    for i,j,z in np.nditer([channelRed, channelGreen, channelBlue]):
        listChannelItem = [i, j, z]
        maxMagVal = max(listChannelItem)
        maxIndex = listChannelItem.index(maxMagVal)
        maxMagValArray.append(maxMagVal)
        maxIndexChannelArray.append(maxIndex)
        
    newMaxMag = np.array(maxMagValArray, dtype=np.float32).reshape((128,64))
    newIndexMaxMag = np.array(maxIndexChannelArray).reshape((128,64))

    maxOrientationValArray = []
    for y, row in enumerate(newIndexMaxMag):
        for x, pixelIndex in enumerate(row):
            # print(y, x, pixelIndex, orientationMAtrix[y,x,pixelIndex])
            maxOrientationValArray.append(orientationMAtrix[y,x,pixelIndex])

    newMaxOrientation = np.array(maxOrientationValArray, dtype=np.float32).reshape((128,64))

    return newMaxMag, newMaxOrientation

def calculate_max_of_3_matrix(matrix1, matrix2, matrix3):
    a = np.maximum(matrix1, matrix2)
    b = np.maximum(a, matrix3)
    return b

def calculate_8_x_8_mag_ori(magMatrix, orientationMAtrix, oneSizeSaquare):
    matrixShape = magMatrix.shape
    subMagArray = []
    subOriArray = []
    for y in range(0,matrixShape[0], oneSizeSaquare):
        for x in range(0,matrixShape[1], oneSizeSaquare):
            subMagArray.append(magMatrix[y: y+oneSizeSaquare, x: x+oneSizeSaquare])
            subOriArray.append(orientationMAtrix[y: y+oneSizeSaquare, x: x+oneSizeSaquare])

    return subMagArray, subOriArray
    
def calculate_all_histogram_in_8_x_8(mag8xArray, ori8x8Array):
    fullGradientList = []
    for matrixMag, matrixOri in zip(mag8xArray, ori8x8Array):
        his = histogram_in_8_x_8(matrixMag,matrixOri)
        fullGradientList.append(his)
    
    imageGradientHist = []
    for i in range(0, len(fullGradientList),8):
        imageGradientHist.append(fullGradientList[i:i+8])

    return imageGradientHist, fullGradientList
    
def calculate_all_histogram_in_8_x_8_optimized(mag8xArray, ori8x8Array):
    fullGradientList = []
    for matrixMag, matrixOri in zip(mag8xArray, ori8x8Array):
        his = histogram_in_8_x_8_optimized(matrixMag,matrixOri)
        fullGradientList.append(his)
    
    imageGradientHist = []
    for i in range(0, len(fullGradientList),8):
        imageGradientHist.append(fullGradientList[i:i+8])

    return imageGradientHist, fullGradientList

def histogram_in_8_x_8(mag8xArray, ori8x8Array):
    matrixShape = mag8xArray.shape
    bin = [0, 20, 40, 60, 80, 100, 120, 140, 160]
    histograBin = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    percentajes = []
    val = []
    for y in range(0,matrixShape[0]):
        for x in range(0,matrixShape[1]):
            valOrientationEvaluated = ori8x8Array[y,x]
            magnitudeEvaluated = mag8xArray[y,x]
            if valOrientationEvaluated in bin:
                # Quiere decir que el valor a buscar es exacto al bin por lo que se pone su monto en una sola casilla
                actualIndex = bin.index(valOrientationEvaluated)                
                finalValue = magnitudeEvaluated + histograBin[actualIndex]
                # print('val orienta ', valOrientationEvaluated, ' val mag ', magnitudeEvaluated, 
                # ' Valor Almacenado en casilla ', histograBin[actualIndex],
                # ' Suma del valor de antes mas el nuevo ', finalValue
                # )
                histograBin[actualIndex] = finalValue       
            else:
                evaluateBinList = [0, 20, 40, 60, 80, 100, 120, 140, 160]
                # tenemos que encontrar su valor anterior y superior segun el bin es dcir si es 10 su anterior es 0 y su superior o posterior es 20
                if valOrientationEvaluated == 180:
                    actualIndex = 0               
                    finalValue = magnitudeEvaluated + histograBin[actualIndex]
                    # print('val orienta ', valOrientationEvaluated, ' val mag ', magnitudeEvaluated, 
                    # ' Valor Almacenado en casilla ', histograBin[actualIndex],
                    # ' Suma del valor de antes mas el nuevo ', finalValue
                    # )
                    histograBin[actualIndex] = finalValue
                else:
                    if valOrientationEvaluated > 160:
                        # Se tiene que repartir entre el 160 y el 0 ya que no xiste el bin de 180
                        previousIndex = 8
                        posteriorIndex = 0

                        porcentageForPreIndex = (180 - valOrientationEvaluated)/20
                        porcentageForPosIndex = (valOrientationEvaluated - 160)/20
                        
                        MagForPreIndex = porcentageForPreIndex * magnitudeEvaluated
                        MagForPosIndex = porcentageForPosIndex * magnitudeEvaluated

                        finalValuePre = MagForPreIndex + histograBin[previousIndex]
                        finalValuePost = MagForPosIndex + histograBin[posteriorIndex]

                        # print('Porcentaje ', porcentageForPreIndex)  
                        # print('val orienta ', valOrientationEvaluated, ' val mag ', magnitudeEvaluated, 
                        # ' Valor Almacenado en casilla "160" ', histograBin[previousIndex],
                        # ' Valor Almacenado en casilla "0" ', histograBin[posteriorIndex], 
                        # ' Valor nuevo en casilla "160" ', MagForPreIndex,
                        # ' Valor nuevo en casilla "0" ', MagForPosIndex,
                        # ' Suma del valor de casilla "160" anterior ', finalValuePre,
                        # ' Suma del valor de casilla "0" anterior ', finalValuePost
                        # )
                        # print('\n\n')

                        histograBin[previousIndex] = finalValuePre
                        histograBin[posteriorIndex] = finalValuePost
                        
                    else:
                        bisect.insort(evaluateBinList, valOrientationEvaluated)
                        posteriorIndex = evaluateBinList.index(valOrientationEvaluated)
                        previousIndex = posteriorIndex - 1

                        porcentageForPreIndex = (bin[posteriorIndex] - valOrientationEvaluated)/20
                        porcentageForPosIndex = (valOrientationEvaluated - bin[previousIndex])/20
                        percentajes.append((porcentageForPreIndex, porcentageForPosIndex))
                        val.append(valOrientationEvaluated)
                        
                        

                        MagForPreIndex = porcentageForPreIndex * magnitudeEvaluated
                        MagForPosIndex = porcentageForPosIndex * magnitudeEvaluated

                        finalValuePre = MagForPreIndex + histograBin[previousIndex]
                        finalValuePost = MagForPosIndex + histograBin[posteriorIndex]
                        
                        # print('Porcentaje ', porcentageForPreIndex)  
                        # print('val orienta ', valOrientationEvaluated, ' val mag ', magnitudeEvaluated, 
                        # ' Valor Almacenado en casilla "160" ', histograBin[previousIndex],
                        # ' Valor Almacenado en casilla "0" ', histograBin[posteriorIndex], 
                        # ' Valor nuevo en casilla "160" ', MagForPreIndex,
                        # ' Valor nuevo en casilla "0" ', MagForPosIndex,
                        # ' Suma del valor de casilla "160" anterior ', finalValuePre,
                        # ' Suma del valor de casilla "0" anterior ', finalValuePost
                        # )
                        # print('\n\n')
                        
                        histograBin[previousIndex] = finalValuePre
                        histograBin[posteriorIndex] = finalValuePost
                
    return histograBin

def histogram_in_8_x_8_optimized(mag8xArray, ori8x8Array):
    matrixShape = mag8xArray.shape
    bin = [0, 20, 40, 60, 80, 100, 120, 140, 160]
    histograBin = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    percentajes = []
    val = []
    for y in range(0,matrixShape[0]):
        for x in range(0,matrixShape[1]):
            valOrientationEvaluated = ori8x8Array[y,x]
            magnitudeEvaluated = mag8xArray[y,x]
            if valOrientationEvaluated in bin:
                # Quiere decir que el valor a buscar es exacto al bin por lo que se pone su monto en una sola casilla
                actualIndex = bin.index(valOrientationEvaluated)                
                finalValue = magnitudeEvaluated + histograBin[actualIndex]
                histograBin[actualIndex] = finalValue       
            else:

                if valOrientationEvaluated == 180:
                    actualIndex = 0               
                    finalValue = magnitudeEvaluated + histograBin[actualIndex]
                    # print('val orienta ', valOrientationEvaluated, ' val mag ', magnitudeEvaluated, 
                    # ' Valor Almacenado en casilla ', histograBin[actualIndex],
                    # ' Suma del valor de antes mas el nuevo ', finalValue
                    # )
                    histograBin[actualIndex] = finalValue
                else:
                    if valOrientationEvaluated > 160:
                        # Se tiene que repartir entre el 160 y el 0 ya que no xiste el bin de 180
                        previousIndex = 8
                        posteriorIndex = 0

                        porcentageForPreIndex = (180 - valOrientationEvaluated)/20
                        porcentageForPosIndex = (valOrientationEvaluated - 160)/20
                        
                        MagForPreIndex = porcentageForPreIndex * magnitudeEvaluated
                        MagForPosIndex = porcentageForPosIndex * magnitudeEvaluated

                        finalValuePre = MagForPreIndex + histograBin[previousIndex]
                        finalValuePost = MagForPosIndex + histograBin[posteriorIndex]

                        histograBin[previousIndex] = finalValuePre
                        histograBin[posteriorIndex] = finalValuePost

                    else:
                        previousIndex = math.floor(valOrientationEvaluated/20)
                        posteriorIndex = math.ceil(valOrientationEvaluated/20)
                        
                        frac, whole = math.modf(valOrientationEvaluated/20)
                        porcentageForPreIndex = 1- frac
                        porcentageForPosIndex = frac
                        percentajes.append((porcentageForPreIndex, porcentageForPosIndex))
                        val.append(valOrientationEvaluated)
                        
                        
                        MagForPreIndex = porcentageForPreIndex * magnitudeEvaluated
                        MagForPosIndex = porcentageForPosIndex * magnitudeEvaluated

                        finalValuePre = MagForPreIndex + histograBin[previousIndex]
                        finalValuePost = MagForPosIndex + histograBin[posteriorIndex]
                                                
                        histograBin[previousIndex] = finalValuePre
                        histograBin[posteriorIndex] = finalValuePost      
    return histograBin

def normalize_gradient_hist_16_x_16(gradientHist):
    """
    Using L2 norm.
    """
    finalNormilizeMatrix = []
    finalNormilizeOneDimension = []
    for y in range(0, 15):
        for x in range(0,7):
            firstSquareTopRight = gradientHist[y][x]
            firstSquareTopLeeft = gradientHist[y][x+1]
            firstSquareBotRightt = gradientHist[y+1][x]
            firstSquareBotLeft = gradientHist[y+1][x+1]
            squareFeatures16x16 = firstSquareTopRight + firstSquareTopLeeft + firstSquareBotRightt + firstSquareBotLeft
            
            k = np.sqrt(np.sum(np.square(squareFeatures16x16)))

            if k == 0:
                numpyArr = np.array(squareFeatures16x16)
                finalNormilizeMatrix.append(numpyArr)
                finalNormilizeOneDimension += squareFeatures16x16     
            else:
                normalizationMatrix = np.divide(squareFeatures16x16,k)
                normalizationList = normalizationMatrix.tolist()
                finalNormilizeMatrix.append(normalizationMatrix)
                finalNormilizeOneDimension += normalizationList
    
    return finalNormilizeMatrix, finalNormilizeOneDimension

def calculate_HOG(imagePath):
    gradX, gradY = calculate_gradient_x_y(imagePath)
    
    mag, ori = calculate_magnitude_orientation_opencv(gradX,gradY)
    # mag1, ori1 = calculate_magnitude_orientation(gradX,gradY)

    newMaxMag, newMaxOri = calculate_max_magnitude_orientation(mag,ori)

    magx8x8Array, ori8x8Array = calculate_8_x_8_mag_ori(newMaxMag, newMaxOri,8)

    imgFullGradientHist, listOfListGradientHist = calculate_all_histogram_in_8_x_8(magx8x8Array,ori8x8Array)

    features, featuresOneDim = normalize_gradient_hist_16_x_16(imgFullGradientHist)
    
    numpyHog3780x1 = np.array(featuresOneDim)

    return features, numpyHog3780x1

def calculate_HOG_optimized(imagePath):
    gradX, gradY = calculate_gradient_x_y(imagePath)
    
    mag, ori = calculate_magnitude_orientation_opencv(gradX,gradY)
    # mag1, ori1 = calculate_magnitude_orientation(gradX,gradY)

    newMaxMag, newMaxOri = calculate_max_magnitude_orientation(mag,ori)

    magx8x8Array, ori8x8Array = calculate_8_x_8_mag_ori(newMaxMag, newMaxOri,8)

    imgFullGradientHist, listOfListGradientHist = calculate_all_histogram_in_8_x_8_optimized(magx8x8Array,ori8x8Array)

    features, featuresOneDim = normalize_gradient_hist_16_x_16(imgFullGradientHist)
    
    numpyHog3780x1 = np.array(featuresOneDim)

    return features, numpyHog3780x1

def calculate_HOG_from_matrix(imgMatrix):
    gradX, gradY = calculate_gradient_x_y_from_matrix(imgMatrix)
    
    mag, ori = calculate_magnitude_orientation_opencv(gradX,gradY)
    # mag1, ori1 = calculate_magnitude_orientation(gradX,gradY)

    newMaxMag, newMaxOri = calculate_max_magnitude_orientation(mag,ori)

    magx8x8Array, ori8x8Array = calculate_8_x_8_mag_ori(newMaxMag, newMaxOri,8)

    imgFullGradientHist, listOfListGradientHist = calculate_all_histogram_in_8_x_8(magx8x8Array,ori8x8Array)

    features, featuresOneDim = normalize_gradient_hist_16_x_16(imgFullGradientHist)
    
    numpyHog3780x1 = np.array(featuresOneDim)

    return features, numpyHog3780x1

def calculate_HOG_from_matrix_optimized(imgMatrix):
    gradX, gradY = calculate_gradient_x_y_from_matrix(imgMatrix)
    
    mag, ori = calculate_magnitude_orientation_opencv(gradX,gradY)
    # mag1, ori1 = calculate_magnitude_orientation(gradX,gradY)

    newMaxMag, newMaxOri = calculate_max_magnitude_orientation(mag,ori)

    magx8x8Array, ori8x8Array = calculate_8_x_8_mag_ori(newMaxMag, newMaxOri,8)

    imgFullGradientHist, listOfListGradientHist = calculate_all_histogram_in_8_x_8_optimized(magx8x8Array,ori8x8Array)

    features, featuresOneDim = normalize_gradient_hist_16_x_16(imgFullGradientHist)
    
    numpyHog3780x1 = np.array(featuresOneDim)

    return features, numpyHog3780x1

def compare_hog_times(imagePath):
    """
    Compere and display the time of execution of all the hog functions.
    """    
    start_time = time.time()
    featuresSquares105, featuresOneList = calculate_HOG(imagePath)
    print("--- %s seconds --- MY HOG L2 ---" % (time.time() - start_time))
    
    start_time = time.time()
    optfeature, optOnefeature = calculate_HOG_optimized(imagePath)
    print("--- %s seconds --- MY HOG optimized L2 ---" % (time.time() - start_time))

    start_time = time.time()
    hogF = hog_using_libraries.calculate_hog_skimage_L2(imagePath)
    print("--- %s seconds --- HOG SKIMAGE L2 ---" % (time.time() - start_time))

    start_time = time.time()
    hogF = hog_using_libraries.calculate_hog_skimage_L2Hys(imagePath)
    print("--- %s seconds --- HOG SKIMAGE L2-Hys ---" % (time.time() - start_time))

    start_time = time.time()
    hogF = hog_using_libraries.calculate_hog_openCV(imagePath)
    print("--- %s seconds --- HOG OPENCV L2-Hys ---" % (time.time() - start_time))

# if __name__ == "__main__":
    # backgroundFolder = r'./dataset/Pedestrians-Dataset-Dummy/Background'
    # pedestrianFolder = r'./dataset/Pedestrians-Dataset-Dummy/Pedestrians'

    # compare_hog_times('./dataset/Pedestrians-Dataset-Dummy/Pedestrians/AnnotationsPos_0.000000_crop001125a_0.png')