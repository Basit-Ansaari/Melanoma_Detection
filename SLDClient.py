from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
from pickle import dump
from pickle import load
from sklearn import svm, metrics, datasets
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split
import skimage
from skimage.io import imread
from skimage.transform import resize
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)
 
"""
    Load image files with categories as subfolder names 
    which performs like scikit-learn sample dataset
    
    Parameters
    ----------
    dir_path : string  
        Path to the main folder holding one subfolder per category
    dimension : tuple
        size to which image are adjusted to (resized)
        
    Returns
    -------
    Bunch
"""



os.system("cls") 

def load_image_files(dir_path, dimension=(64, 64)):
    image_dir = Path(dir_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]

    descr = "Skin Lesion - Malanoma Dataset"
    images = []
    flat_data = []
    target = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            img = imread(file)
            img = skimage.io.imread(file)
            img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
            flat_data.append(img_resized.flatten()) 
            images.append(img_resized)
            target.append(i)
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)
    return Bunch(data=flat_data,
                 target=target,
                 target_names=categories,
                 images=images,
                 DESCR=descr)

def load_unseen_image(filename, dimension=(64, 64)): 
    descr = "Unseen Dataset"	
    images = []
    flat_data = []
    #target = []
    img = imread(filename)
    img = skimage.io.imread(filename)
    img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
    flat_data.append(img_resized.flatten()) 
    images.append(img_resized)
    #target.append(i)
    flat_data = np.array(flat_data)
    #target = np.array(target)
    images = np.array(images)
    return Bunch(data=flat_data, images=images, DESCR=descr)     


print('Loading Images...')
image_dataset = load_image_files("images/")
print('Loaded.')
# dump(image_dataset, open('image_dataset.pkl', 'wb'))

# Split data
print('Split Training and Testing data...')
X_train, X_test, y_train, y_test = train_test_split(image_dataset.data, image_dataset.target, test_size=0.2,random_state=109)
print('\nX_Test')
print(X_test)
print('\nX_Test[0]')
print(X_test[0])
print('\nX_Test Length')
print(len(X_test))
print('\nData[0]')
print(image_dataset.data[0])
print('\nTargrt[0]')
print(image_dataset.target[0])
print('----------------------------------------------')
print('\nTargets')
print(image_dataset.target)
print('----------------------------------------------')
print('\nTarget Length')
print(len(image_dataset.target))
print('----------------------------------------------')
print('\nTarget Names')
print(image_dataset.target_names)

"""
# Train data with parameter optimization
param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear','poly']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['poly','rbf']},
 ]
 
svc = svm.SVC()

clf = GridSearchCV(estimator=svc,cv=10,param_grid=param_grid,refit=True)
print('Creating GridSearchCV Model with cv = 3') #####################################################################
model = clf.fit(X_train, y_train)
print('Model created')

print('\nGridSearchCV Model')
print(clf)

# Save the model to disk
filename = 'Melanoma.pkl'
print('Saving Trained Model to ' + filename)
dump(model, open(filename, 'wb'))
print('Trained Model saved.') 
"""

# Load the model from disk
filename = 'Melanoma.pkl'
print('\nLoading Trained Model from Melanoma.pkl') 
loaded_model = load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print('Accuracy (Loaded Model) : ', result)

# Predict
print('Predicting...')
y_pred = loaded_model.predict(X_test)
print('Done')
print('\nPredictions')
print(y_pred)
print('\nPrediction Length : ',len(y_pred))

# Save the model to disk
filename = 'Income.pkl'
print('Saving Trained Model to ' + filename)
dump(model, open(filename, 'wb'))
print('Trained Model saved.') 

# Report
print("\nClassification Report")
print(metrics.classification_report(y_test, y_pred))
print('\nY_test')
print(y_test, len(y_test))
print('\nY_Pred')
print(y_pred, len(y_pred))

accuracy = accuracy_score(y_test, y_pred)	
print("\nAccuracy Score: ", accuracy)

print('Total Images : ', len(image_dataset.target))
print('Total Predictions : ', len(y_pred))

print('\nManual Prediction with X_test')
y_pred2 = loaded_model.predict(X_test[0].reshape(1, -1))
print('\nPredictions II')
print(y_pred2)

y_pred3 = loaded_model.predict(X_test[397].reshape(1, -1))
print('\nPredictions III')
print(y_pred3)

y_pred4 = loaded_model.predict(X_test[190].reshape(1, -1))
print('\nPredictions IV')
print(y_pred4)

"""
for i in range(len(X_test)):
    if y_pred[i] == 1:
        print(i, 1)
"""
lst = [187,201,214,244,397]
for i in lst:
    y_predN = loaded_model.predict(X_test[i].reshape(1, -1))
    print('\nPredictions V')
    print(y_predN)

# images/Benign/ISIC_0000001.jpg
unseen = load_unseen_image('images/Benign/ISIC_0000001.jpg')
#print(len(unseen.data))
print('ISIC_0000001.jpg (Transformed) : ', unseen.data)

# Predicting loaded image (Benign)
y_predUnseen = loaded_model.predict(unseen.data.reshape(1, -1))
print('\n\nPredictions unseen loaded image')
print(y_predUnseen)

# images/Malignant/ISIC_0000029.jpg

# Predicting loaded image (Malignant)
unseen2 = load_unseen_image('images/Malignant/ISIC_0000029.jpg')
print('ISIC_0000029.jpg (Transformed) : ', unseen2.data)
y_predUnseen2 = loaded_model.predict(unseen2.data.reshape(1, -1))
print('\n\nPredictions unseen loaded image')
print(y_predUnseen2)


