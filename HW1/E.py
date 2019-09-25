import matplotlib.pyplot as plt
import arff
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

xValue = []
labelValue = []

for row in arff.load('column_2C_weka.arff'):
    xValue.append([row[0],row[1],row[2],row[3],row[4],row[5]])
    labelValue.append(row[6])

trainingSetX = xValue[0:140] + xValue[210:280]
trainingSetLabel = labelValue[0:140] + labelValue[210:280]
testSetX = xValue[140:210] + xValue[280:310]
testSetLabel = labelValue[140:210] +labelValue[280:310]


# manhattan metric
manTestError = []
manTrainingError = []
manIndex = []

for i in range(1,197,5):
    knn = KNeighborsClassifier(n_neighbors=i,p=1,weights='distance')
    knn.fit(trainingSetX,trainingSetLabel)
    testScore = knn.score(testSetX,testSetLabel)
    manTestError.append(1 - testScore)
    trainingScore = knn.score(trainingSetX, trainingSetLabel)
    manTrainingError.append(1 - trainingScore)
    manIndex.append(i)

print('manhattan metric:')
k = manIndex[manTestError.index(min(manTestError))]
print('best k = ',k)
print('error rate = ',min(manTestError))
print('')

plt.title('manhattan test error rate')
plt.xlabel('K')
plt.ylabel('error rate')
plt.plot(manIndex,manTrainingError,color = 'blue', label = 'trainingError')
plt.plot(manIndex,manTestError,color = 'red', label = 'testError')
plt.legend()
plt.savefig("manhattan test error in distance.png")
plt.show()





# Euclidean metric
eucTestError = []
eucTrainingError = []
eucIndex = []

for i in range(1,197,5):
    knn = KNeighborsClassifier(n_neighbors=i,weights='distance')
    knn.fit(trainingSetX,trainingSetLabel)
    testScore = knn.score(testSetX,testSetLabel)
    eucTestError.append(1 - testScore)
    trainingScore = knn.score(trainingSetX, trainingSetLabel)
    eucTrainingError.append(1 - trainingScore)
    eucIndex.append(i)
print('Euclidean metric:')
print('best k =',eucIndex[eucTestError.index(min(eucTestError))])
print('test error = ',min(eucTestError))
print('')

plt.title('Euclidean test error rate')
plt.xlabel('K')
plt.ylabel('test error')
plt.plot(eucIndex,eucTrainingError,color = 'blue', label = 'trainingError')
plt.plot(eucIndex,eucTestError,color = 'red', label = 'testError')
plt.legend()
plt.savefig("Euclidean test error in distance.png")
plt.show()


# chebyshev metric
chebyTestError = []
chebyTrainingError = []
chebyIndex = []

for i in range(1,197,5):
    knn = KNeighborsClassifier(n_neighbors=i,metric='chebyshev',weights='distance')
    knn.fit(trainingSetX,trainingSetLabel)
    testScore = knn.score(testSetX,testSetLabel)
    chebyTestError.append(1 - testScore)
    trainingScore = knn.score(trainingSetX, trainingSetLabel)
    chebyTrainingError.append(1 - trainingScore)
    chebyIndex.append(i)

k = chebyIndex[chebyTestError.index(min(chebyTestError))]
print('chebyshev metric:')
print('best k = ',k)
print('test error = ',min(chebyTestError))
print('')

plt.title('chebyshev test error rate')
plt.xlabel('K')
plt.ylabel('error rate')
plt.plot(chebyIndex,chebyTrainingError,color = 'blue', label = 'trainingError')
plt.plot(chebyIndex,chebyTestError,color = 'red', label = 'testError')
plt.legend()
plt.savefig("chebyshev test error in distance.png")
plt.show()
