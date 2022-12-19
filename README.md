
# Detection of Anomalous Images

Detecting anomalous images from a large dataset of facial images of human subjects primarily using statistical and probablistic methods.

## Get Started

#### Dependency

- Numpy
- Pandas
- cv2
- scikit-learn
- mahotas
- pywt
- hdbscan
- scikit-image
- matplotlib
- xgboost

#### Data Preperation

Create an instance with four datasets as inputs. The paths for the good training data, anomalous training data, good testing data, and anomalous testing data are to be provided.

```python
# Initiate an instance 
a1 = Anomdet()
```

#### Feature Selection and Extraction

Depending on the method we want to use wether a simple SVDD/GLOSH or an ensemble we can choose the set of features to be extracted. The Array A is used as a feature extractor and a 1 indicates that the corresponding feature is selected while a 0 indicated the opposite.

The features are in the order from A[0] to A[n] and additional features can be extended with similar notation

Feature Extraction is done for each of the dataset and individually and then converted into a data frame which can be used to train or test the classifiers

```python
# Feature selector given in the form of an array

# select 4 different features to extract from a specific data set
a1.getdata("fin_data/good/",[1,1,1,0,1])


# select 1 different features to extract, this is mainly used for ensemble cases
a1.getdata("fin_data/good/",[0,1,0,0,0])

```


## Training and obtaining predictions

Depending on the model that needs to be implemented we will fit the model by calling the corresponding function. The ```alltrain_features``` variable is the dataframe of training images obtained from the above step

```python
# To train a SVDD with the predefined dataset
a1.svdtrain(alltrain_features)

# To fit the HDBSCAN with the training data
a1.clusters.fit(alltrain_features)
```

Once the model is fitted testing is done on the fitted model. The testing data can be of any configuration when implementing a simple classifier. But in the case of ensemble implementation its better to have 50% anomaly rate datasets for better performance. 

```python
# Predicting with the test features
temp = []
temp = a1.svm_model.predict(a1.alltest_features)
# Since outliers are labeled as -1 in svdd and glosh
pred1 = [0 if i==-1 else 1 for i in pred]

# actual labels for the images 
truth = np.concatenate((np.ones((a1.n_of_goodt, )), np.zeros((a1.n_of_badt, ))), axis=0)
```
## Ensemble
A new instance is initiated for the ensemble class function.

The predictions from the individual classifiers are used as features while the truth variable is used as a label for the ensemble model.

Training and testing the ensemble is the same as it is for the classifier and once the model acheives adequate performance the fitted model can be used predict the anomalous nature of the images.

```python
#Creating an instance of EnsembleMod
e1 = EnsmebleMod(ensfeat, truth)

# XGBoost ensemble
e1.xgbr()

# Bagging ensemble
e1.bagg()

# Multilayer perceptron 
e1.mlpc()

# To predict anomalous images in a dataset Y we generate the feature dataframe using the feature extraction step and use that dataframe to predict.
prediction = e1.clf.predict(Y)
```

