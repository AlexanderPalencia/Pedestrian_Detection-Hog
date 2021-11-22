# Manuel Alexander Palencia Gutierrez
import numpy as np
import pandas as pd
import itertools

# link reference
# https://towardsdatascience.com/non-maxima-suppression-139f7e00f0b5


def read_persisted_dataframe(path):
    return pd.read_pickle(path)


def NMS_score(pboxes, probaility, overlapThresh = 0.4):
    # Order coordinates
    originalProb = probaility.copy()
    probaility.sort(reverse=True)

    newBoxex = []
    for i, item in enumerate(probaility):
        index = originalProb.index(item)
        newBoxex.append(pboxes[index])

    boxes = np.array(newBoxex)
    
    # Return an empty list, if no boxes given
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]  # x coordinate of the top-left corner
    y1 = boxes[:, 1]  # y coordinate of the top-left corner
    x2 = boxes[:, 2]  # x coordinate of the bottom-right corner
    y2 = boxes[:, 3]  # y coordinate of the bottom-right corner
    # Compute the area of the bounding boxes and sort the bounding
    # Boxes by the bottom-right y-coordinate of the bounding box
    areas = (x2 - x1 + 1) * (y2 - y1 + 1) # We add 1, because the pixel at the start as well as at the end counts
    # The indices of all boxes at start. We will redundant indices one by one.
    indices = np.arange(len(x1))

    finalCor = []
    finalSquareIndex = []
    for i,box in enumerate(boxes):
        # Create temporary indices  
        temp_indices = indices[indices!=i]
        # Find out the coordinates of the intersection box
        xx1 = np.maximum(box[0], boxes[temp_indices,0])
        yy1 = np.maximum(box[1], boxes[temp_indices,1])
        xx2 = np.minimum(box[2], boxes[temp_indices,2])
        yy2 = np.minimum(box[3], boxes[temp_indices,3])

        # Find out the width and the height of the intersection box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / areas[temp_indices]

        condition = overlap > overlapThresh
        # if the actual boungding box has an overlap bigger than treshold with any other box, remove it's index  
        if np.any(overlap > overlapThresh):
            # The coordinate that pass the treshold is i this index is what we take away
            
            squaresInsideSquareIndex = [i]
            # Square that we are validating or checking
            for index2, bol in enumerate(condition):
                if bol == True or bol == "True":
                    squaresInsideSquareIndex.append(temp_indices[index2])
            finalSquareIndex.append(squaresInsideSquareIndex)

    print(finalSquareIndex)
    FinalfinalBoxesIndex = []
    for listIndex in finalSquareIndex:
        searchVal = listIndex[0]
        listFind = []
        for a in finalSquareIndex:
            if searchVal in a:
                listFind.append(a)
                if len(listFind)>1:
                    lenMayor = 0
                    for inde, b in enumerate(listFind):
                        if len(b) > lenMayor:
                            lenMayor = len(b)
                            mayorList = b

                                
                    if mayorList in FinalfinalBoxesIndex:
                        pass
                    else:
                        FinalfinalBoxesIndex.append(mayorList)
                    
                else:
                    FinalfinalBoxesIndex.append(a)

    finalFinalFinalindex = []
    # No dupllicados
    for listEval in FinalfinalBoxesIndex:
        if listEval in finalFinalFinalindex:
            pass
        else:
            finalFinalFinalindex.append(listEval)


    print('no dup ', finalFinalFinalindex)

    lastMaxPriba = []
    for listInList in finalFinalFinalindex:
        probaSquare = []
        for indexFinal in listInList:
            # calculando probabilidad
            probaSquare.append(probaility[indexFinal])
        IndexMaxProba = probaSquare.index(max(probaSquare))
        lastMaxPriba.append(listInList[IndexMaxProba])
    print('ultimos inde ', lastMaxPriba)

    finalBoxes = []
    finalProbabiliy = []
    for k in lastMaxPriba:
        finalBoxes.append(boxes[k])
        finalProbabiliy.append(probaility[k])
        
    npFinalBoxes = np.array(finalBoxes)

    print('coorr ', npFinalBoxes)

        #     newIndexPro = probaility.index(max(squreProba))
        #     # Que cornedas se queda? se queda la mayor
        #     finalCor.append(boxes[newIndexPro])
            
            # indices = indices[indices != i]


    
    #return only the boxes at the remaining indices
    return npFinalBoxes.astype(int), finalProbabiliy



def NMS_no_score(boxes, overlapThresh = 0.4):
    # Return an empty list, if no boxes given
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]  # x coordinate of the top-left corner
    y1 = boxes[:, 1]  # y coordinate of the top-left corner
    x2 = boxes[:, 2]  # x coordinate of the bottom-right corner
    y2 = boxes[:, 3]  # y coordinate of the bottom-right corner
    # Compute the area of the bounding boxes and sort the bounding
    # Boxes by the bottom-right y-coordinate of the bounding box
    areas = (x2 - x1 + 1) * (y2 - y1 + 1) # We add 1, because the pixel at the start as well as at the end counts
    # The indices of all boxes at start. We will redundant indices one by one.
    indices = np.arange(len(x1))
    for i,box in enumerate(boxes):
        # Create temporary indices  
        temp_indices = indices[indices!=i]
        # Find out the coordinates of the intersection box
        xx1 = np.maximum(box[0], boxes[temp_indices,0])
        yy1 = np.maximum(box[1], boxes[temp_indices,1])
        xx2 = np.minimum(box[2], boxes[temp_indices,2])
        yy2 = np.minimum(box[3], boxes[temp_indices,3])

        # Find out the width and the height of the intersection box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / areas[temp_indices]
        # if the actual boungding box has an overlap bigger than treshold with any other box, remove it's index  
        if np.any(overlap > overlapThresh):
            indices = indices[indices != i]
    #return only the boxes at the remaining indices
    return boxes[indices].astype(int)




def NMS_Class(candidates, overlap_threshold=0.5):
    if len(candidates) < 2:
        return candidates
    
    if candidates.dtype.kind == "i":
	    candidates = candidates.astype("float")

    non_supressed_boxes = []

    # grab the coordinates of the bounding boxes
    x1 = candidates[:,0]
    y1 = candidates[:,1]
    x2 = candidates[:,2]
    y2 = candidates[:,3]
 
 	# compute the area of the bounding boxes w*h
    box_areas = candidates[:,2]*candidates[:,3]
    idxs = np.argsort(y2)
    
    print('array sort ', idxs)
    selected = []
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        last_idx = idxs[last]
        selected.append(last_idx)
        
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[last_idx], x1[idxs[:last]])
        yy1 = np.maximum(y1[last_idx], y1[idxs[:last]])
        xx2 = np.minimum(x2[last_idx], x2[idxs[:last]])
        yy2 = np.minimum(y2[last_idx], y2[idxs[:last]])
        
        
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)    
  
  		# compute the ratio of overlap
        overlap = (w * h) / box_areas[idxs[:last]]

        print(' Square ', overlap)
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],np.where(overlap > overlap_threshold)[0])))
    
    s = len(selected)-len(candidates)
    print('Este valor es S ', s)
    if s < 0:
        print(f'Suppressed candidates: {s}')
        
    return candidates[selected].astype("int")





# if __name__ == "__main__":
#     # df = read_persisted_dataframe('./results/test_train/df_pyramid_test_train.pkl')
#     # listcoordinates = df['position_corner_xy1_xy2'].to_list()

#     arraycoordinates = np.array([[220, 206, 162, 162]
#                   ,[230, 190, 162, 162]
#                   ,[240,230,162,162]
#                   ,[10,20,162,162]
#                   ,[280,280,40,40]
#                   ])
#     print(len(arraycoordinates))



#     newCoordinates = NMS(arraycoordinates, 0.5)
#     print('Nuevas cordenadas')
#     for coordinate in newCoordinates:
#         topNewLeftCorner = tuple(coordinate[0:2])
#         rigthNewBoCorner = tuple(coordinate[2:])
#         print('Coor 1 ',topNewLeftCorner, ' Coor 2',rigthNewBoCorner)

#     print('virgen')
#     for (x,y,w,h) in NMS_Class(arraycoordinates):
#         print('Coor 1 ', (x,y), ' Coor 2',(w,h))
