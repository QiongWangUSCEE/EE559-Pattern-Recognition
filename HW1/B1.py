import matplotlib.pyplot as plt
import arff

plt.rcParams['figure.figsize'] = (24, 18)
plt.rcParams['savefig.dpi'] = 300
plt.figure()


for i in range (0,6):
    for j in range (0,6):
        plt.subplot(6,6,i*6+j+1)
        if i == j == 0:
            plt.text(0.5,0.5,'pelvic incidence',fontsize = 18,verticalalignment="center",horizontalalignment="center")
            continue
        if i == j == 1:
            plt.text(0.5,0.5,'pelvic tilt',fontsize = 18,verticalalignment="center",horizontalalignment="center")
            continue
        if i == j == 2:
            plt.text(0.5,0.5,'lumbar lordosis angle',fontsize = 18,verticalalignment="center",horizontalalignment="center")
            continue
        if i == j == 3:
            plt.text(0.5,0.5,'sacral slope',fontsize = 18,verticalalignment="center",horizontalalignment="center")
            continue
        if i == j == 4:
            plt.text(0.5,0.5,'pelvic radius',fontsize = 18,verticalalignment="center",horizontalalignment="center")
            continue
        if i == j == 5:
            plt.text(0.5,0.5,'grade of spondylolisthesis',fontsize = 18,verticalalignment="center",horizontalalignment="center")
            continue

        xValue1 = []
        yValue1 = []
        xValue2 = []
        yValue2 = []

        for row in arff.load('column_2C_weka.arff'):
            if row[6] == "Abnormal":
                xValue1.append(row[i])
                yValue1.append(row[j])
            else:
                xValue2.append(row[i])
                yValue2.append(row[j])

        plt.scatter(xValue1,yValue1,c="red",marker="o")

        plt.scatter(xValue2,yValue2,c="blue",marker="x")

plt.savefig("scatterplots.png")

plt.show()


