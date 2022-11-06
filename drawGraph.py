import matplotlib.pyplot as plt
import numpy as np
testData = []
trainData = []
costData = []
path = input("Insert path: ")
with open(path + "/testData.txt") as f:
    for line in f:
        
        editedLine = line.split(', ')
        editedLine.pop()
        for x in editedLine:
            testData.append(float(x))

with open(path + "/trainData.txt") as f:
    for line in f:
        editedLine = line.split(', ')
        editedLine.pop()
        for x in editedLine:
            trainData.append(float(x))

with open(path + "/costData.txt") as f:
    for line in f:
        editedLine = line.split(', ')
        editedLine.pop()
        for x in editedLine:
            costData.append(float(x)*10)

xaxis = []

for elem_idx in range(len(trainData)):
    xaxis.append(round(elem_idx/2, 4))

listof_YTicks = np.arange(0, 100, 5)
plt.yticks(listof_YTicks)

plt.plot(xaxis, trainData, label="Train accuracy")
plt.plot(xaxis, testData, label="Test accuracy")
plt.plot(xaxis, costData, label="Cost accuracy (x10)")
plt.xlabel("Epochs")
plt.ylabel("Accuracy %")
plt.legend()
plt.show()
