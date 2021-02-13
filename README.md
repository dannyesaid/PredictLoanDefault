## Multilayer perceptron neural network that predicts if an applicant will default on a loan. Uses Kaggle dataset https://www.kaggle.com/ajay1735/hmeq-data

```python
import pandas
import numpy
import tensorflow
import matplotlib.pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout

#load the csv file and rename the columns
dataset = pandas.read_csv( 'LoanDefaultData.csv' )

dataset.rename( { 'BAD': 'defaultResult', 'LOAN': 'loanRequested', 'MORTDUE': 'currentMortageDue',\
                  'VALUE': 'currentPropertyValue', 'REASON': 'reasonForRequest', 'JOB': 'Job',\
                  'YOJ': 'yearsAtJob', 'DEROG': 'derogatoryReports', 'DELINQ': 'delinquentLines',\
                  'CLAGE': 'ageOldestTradeLine', 'NINQ': 'recentCreditLines', 'CLNO': 'totalCreditLines',\
                  'DEBTINC': 'debtToIncome' }, inplace = True, axis = 'columns' )

#shuffle the dataframe
loanDefaultDataset.reindex( numpy.random.permutation( loanDefaultDataset.index ) );
```


```python

#replace nan values with averages and modes
dataset.currentMortageDue.fillna( dataset.currentMortageDue.mean(), inplace = True )

dataset.currentPropertyValue.fillna( dataset.currentPropertyValue.mean(), inplace = True )

dataset.reasonForRequest.fillna( dataset.reasonForRequest.mode().iloc[ 0 ] , inplace = True )

dataset.Job.fillna( dataset.Job.mode().iloc[ 0 ], inplace = True )

dataset.yearsAtJob.fillna( dataset.yearsAtJob.mean(), inplace = True )

dataset.derogatoryReports.fillna( dataset.derogatoryReports.mode().iloc[ 0 ], inplace = True )

dataset.delinquentLines.fillna( loanDefaultDataset.delinquentLines.mode().iloc[ 0 ], inplace = True )

dataset.ageOldestTradeLine.fillna( dataset.ageOldestTradeLine.mean(), inplace = True )

dataset.recentCreditLines.fillna( dataset.recentCreditLines.mode().iloc[ 0 ], inplace = True )

dataset.totalCreditLines.fillna( dataset.totalCreditLines.mode().iloc[ 0 ], inplace = True )

dataset.debtToIncome.fillna( dataset.debtToIncome.mean(), inplace = True )

#one hot encode the dataframe
dataset = pandas.get_dummies( dataset )
```


```python
#normalize the data
datasetColumns = dataset.columns

SklearnMinMaxScaler = MinMaxScaler( copy = False )

SklearnMinMaxScaler.fit( dataset )

dataset = SklearnMinMaxScaler.transform( dataset )

dataset = pandas.DataFrame( dataset, columns = datasetColumns );


```


```python
#split the data into train, validation and test sets
dataSetTargets = dataset[ 'defaultResult' ]

dataSetFeatures = dataset.drop( [ 'defaultResult' ], axis = 'columns' )

dataSetTargetsTrain = dataSetTargets[ :3500 ]

dataSetTargetsValidation = dataSetTargets[ 3500:4600 ]

dataSetTargetsTest =  dataSetTargets[ 4600: ]

dataSetFeaturesTrain = dataSetFeatures[ :3500 ]

dataSetFeaturesValidation = dataSetFeatures[ 3500:4600 ]

dataSetFeaturesTest =  dataSetFeatures[ 4600: ]

```


```python
#find best parameters for binary classifier
binaryClassifier = Sequential()

binaryClassifier.add( Dense( 64, activation = 'relu', input_shape = ( 18, ) ) )
binaryClassifier.add( Dropout( 0.55 ) )
binaryClassifier.add( Dense( 64, activation = 'relu' ) )
binaryClassifier.add( Dropout( 0.55 ) )
binaryClassifier.add( Dense( 1, activation = 'sigmoid' ) )


binaryClassifier.compile(optimizer = 'rmsprop', 
                         loss = 'binary_crossentropy', 
                         metrics = [ 'binary_accuracy' ])


binaryClassifierHistory = binaryClassifier.fit(loanDefaultDataSetFeaturesTrain, 
                         loanDefaultDataSetTargetsTrain, 
                         batch_size = 128, 
                         epochs = 100,
                         validation_data = ( loanDefaultDataSetFeaturesValidation, loanDefaultDataSetTargetsValidation ),
                         verbose = False);


binaryClassifierHistory = binaryClassifierHistory.history
```


```python
#plot validation accuracy
validationBinaryAccuracy = binaryClassifierHistory[ 'val_binary_accuracy' ] 


matplotlib.pyplot.plot( range( 100 ), validationBinaryAccuracy );



```


![png](Predict%20loan%20default_files/Predict%20loan%20default_5_0.png)



```python

#train classifier with best parameters found on all train data
featuresTrainFinal = pandas.concat([dataSetFeaturesTrain,
                                    dataSetFeaturesValidation],
                                    axis = 0)

targetsTrainFinal = pandas.concat([ dataSetTargetsTrain, 
                                    dataSetTargetsValidation],
                                    axis = 0)


historyFinal = binaryClassifier.fit(featuresTrainFinal, 
                                    targetsTrainFinal, 
                                    batch_size = 128, 
                                    epochs = 50,
                                    validation_data = ( dataSetFeaturesTest, dataSetTargetsTest ),
                                    verbose = False);


historyFinal = historyFinal.history

      
    
```


```python
#output final accuracy
print( 'final binary accuracy:', historyFinal[ 'val_binary_accuracy' ][49] )

predictions = binaryClassifier.predict( dataSetFeaturesTest )

print()
print()
print()

defaults = [ element[0] for element in predictions if element >= 0.5 ]

nonDefaults = [ element[0] for element in predictions if element < 0.5 ]

print('default predictions')

for default in defaults:
    print(default)

print()
print()
print()
print('non default predictions')

for nonDefault in nonDefaults:
    print(nonDefault)



```

    final binary accuracy: 0.851470587534063
    
```
