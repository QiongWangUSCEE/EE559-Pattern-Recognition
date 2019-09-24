import matplotlib.pyplot as plt
import arff
import math
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

subsetX = []
subsetLabel = []
indexN = []
minError = []

for N in range (10,220,10):
    subsetX = trainingSetX[0:N-math.floor(N/3)] + trainingSetX[140:140 + math.floor(N/3)]
    subsetLabel = trainingSetLabel[0:N-math.floor(N/3)] + trainingSetLabel[140:140 + math.floor(N/3)]
    indexN.append(N)

    testError = []

    for i in range(1, N, 2):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(subsetX, subsetLabel)
        testScore = knn.score(testSetX, testSetLabel)
        testError.append(1 - testScore)


    minError.append(min(testError))

plt.plot(indexN,minError)
plt.title('best test error rate against N curve')
plt.xlabel('N')
plt.ylabel('best test error rate')
plt.savefig("best test error rate against N.png")
plt.show()


