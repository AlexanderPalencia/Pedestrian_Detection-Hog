import pandas as pd
from sklearn.svm import SVC
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# https://scikit-learn.org/stable/modules/svm.html#scores-probabilities
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC.predict

def read_persisted_dataframe(path):
    return pd.read_pickle(path)

def train_model_svm(trainData, labelsTrain):
    svm_model = SVC(random_state=42, tol=1e-5)
    return svm_model.fit(trainData, labelsTrain)

def save_model_pickle(filenameModel, modelToSave):
    # save the model to disk
    filename = filenameModel
    pickle.dump(modelToSave, open(filename, 'wb'))
    return 'Successfully Saved'

def load_pickle_model(filenameModel):
    # some time later...
    # load the model from disk
    loaded_model = pickle.load(open(filenameModel, 'rb'))
    return loaded_model

def pipeline_to_train(pathDataFrame, pathNameSVM):
    print('Training SVM model with {}'.format(pathDataFrame))
    df = read_persisted_dataframe(pathDataFrame)
    cleanDf = df[df['label'] != -1]
    # Read and transform dataframe columns
    hogFeatures = cleanDf['HOG_Features'].to_list()
    labels = cleanDf['label'].to_list()

    X_train, X_test, y_train, y_test = train_test_split(hogFeatures, labels, test_size=0.05, random_state=42)

    smvTrained = train_model_svm(X_train, y_train)
    print('Successfully training SVM model')
    save_model_pickle(pathNameSVM, smvTrained)



def pipeline_get_score(pathDataFrame, pathNameSVM):
    print('Scores {}'.format(pathDataFrame))
    df = read_persisted_dataframe(pathDataFrame)
    cleanDf = df[df['label'] != -1]
    # Read and transform dataframe columns
    hogFeatures = cleanDf['HOG_Features'].to_list()
    labels = cleanDf['label'].to_list()

    X_train, X_test, y_train, y_test = train_test_split(hogFeatures, labels, test_size=0.05, random_state=42)

    # load models
    model = load_pickle_model(pathNameSVM)
    
    scoreee = model.score(X_test, y_test)

    actualPrediction = [list(y_test)]
    prediction = [model.predict(X_test)]

    print(classification_report(actualPrediction, prediction))

    # print('Successfully Score')
    print(scoreee)


if __name__ == "__main__":

    arrayDataFrame = ['./persisted_data/df_hog_opencv.pkl', './persisted_data/df_hog_skimage_l2_hys.pkl', './persisted_data/df_hog_skimage_l2.pkl', './persisted_data/df_my_hog_function.pkl']

    arraySavModel = ['./persisted_data/svm_opencv.sav', './persisted_data/svm_skimage_l2_hys.sav', './persisted_data/svm_skimage_l2.sav', './persisted_data/svm_my_hog.sav']

    for dfName, savName in zip(arrayDataFrame, arraySavModel):
        pipeline_get_score(dfName, savName)
    