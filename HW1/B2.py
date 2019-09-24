import matplotlib.pyplot as plt
import arff

plt.rcParams['figure.figsize'] = (10, 5)
plt.figure()
xLabels = ['AB','NO']
yLabels = ['pelvic incidence','pelvic tilt','lumbar lordosis angle','sacral slope','pelvic radius','grade of spondylolisthesis']
for i in range (0,6):
    plt.subplot(1,6,i+1)

    abValue1 = []
    noValue2 = []

    for row in arff.load('column_2C_weka.arff'):
        if row[6] == "Abnormal":
            abValue1.append(row[i])
        else:
            noValue2.append(row[i])

    plt.boxplot([abValue1, noValue2],labels = xLabels)
    plt.ylabel(yLabels[i])

plt.savefig("boxplot.png")

plt.show()