import matplotlib.pyplot as plt
import arff
import numpy as np
import itertools
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

xValue = []
labelValue = []

for row in arff.load('column_2C_weka.arff'):
    xValue.append([row[0],row[1],row[2],row[3],row[4],row[5]])
    labelValue.append(row[6])

trainingSetX = xValue[0:140] + xValue[210:280]
trainingSetLabel = labelValue[0:140] + labelValue[210:280]
testSetX = xValue[140:210] + xValue[280:310]
testSetLabel = labelValue[140:210] +labelValue[280:310]

testError = []
trainingError = []
index = []

for i in range(1,210,2):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(trainingSetX,trainingSetLabel)
    testScore = knn.score(testSetX,testSetLabel)
    testError.append(1 - testScore)
    trainingScore = knn.score(trainingSetX, trainingSetLabel)
    trainingError.append(1 - trainingScore)
    index.append(i)

print('best k = ',index[testError.index(min(testError))])
print('error rate = ',min(testError))

plt.xlabel('K')
plt.ylabel('test error')
plt.plot(index,trainingError,color = 'blue', label = 'trainingError')
plt.plot(index,testError,color = 'red', label = 'testError')
plt.legend()
plt.title('test error curve')
plt.savefig("test error curve.png")
plt.show()

knn1 = KNeighborsClassifier(n_neighbors=index[testError.index(min(testError))])
knn1.fit(trainingSetX,trainingSetLabel)
confusionMatrix = confusion_matrix(testSetLabel,knn1.predict(testSetX))

plt.imshow(confusionMatrix)
labels = ['AB','NO']
xlocation = np.array(range(len(labels)))
plt.xticks(xlocation,labels,rotation=0)
plt.yticks(xlocation,labels)
plt.title('Confusion Matrix')
plt.xlabel('true test label')
plt.ylabel('predict test label')
plt.colorbar()

for i,j in itertools.product(range(confusionMatrix.shape[0]),range(confusionMatrix.shape[1])):
    plt.text(j,i,confusionMatrix[i,j],horizontalalignment='center')

plt.savefig("confusion Matrix.png")
plt.show()






