import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random
import time

FIRST_GROUP = 0
SECOND_GROUP = 1
X_COORDINATE = 0
Y_COORDINATE = 1
SAMPLE_SIZE = 32  # I wish to change this along the way
groups = [[], []]
GROUP_SIZE = 32
NUM_ITERATIONS = 40
LEARNING_RATE = 0.07  # that to o
VECTORS_TO_SHOW = min(20, NUM_ITERATIONS)
HISTORY_TO_SHOW = min(NUM_ITERATIONS, 1000)

# Create groups
for _ in range(GROUP_SIZE):
    x = random.uniform(-3.5, 0.5)
    y = random.uniform(-0.5, 0.2)
    groups[FIRST_GROUP].append([x, y])
for _ in range(GROUP_SIZE):
    x = random.uniform(0.6, 1.4)
    y = random.uniform(1.2, 2)
    groups[SECOND_GROUP].append([x, y])

# groups[FIRST_GROUP] = [[-1, -1], [-0.2, -0.2]]
# groups[SECOND_GROUP] = [[0.5, 0.5], [2, 2]]
points = [np.array(groups[FIRST_GROUP]), np.array(groups[SECOND_GROUP])]

"""
distance of functon y=ax+b for point px,py
side - first group needs to be bellow vector,second group above vector
"""


def Distance(m, n, px, py, side):
    return max(0, side * (m * px - py + n) / ((m ** 2 + 1) ** 0.5))  # (m*px+n-py)*side)#


"""
Loss Sum (= distance sum)
"""

'''
def LossSum(a, b, ps):
    sum = 0
    for i in range(len(ps[FIRST_GROUP])):
        sum += Distance(a, b, ps[FIRST_GROUP][i][X_COORDINATE], ps[FIRST_GROUP][i][Y_COORDINATE], 1)
    for i in range(len(ps[SECOND_GROUP])):
        sum += Distance(a, b, ps[SECOND_GROUP][i][X_COORDINATE], ps[SECOND_GROUP][i][Y_COORDINATE], -1)
    return sum'''


"""
new a or new b caluted based on gradient
y is the loss
"""




"""
Get sample from a group
"""


def GetSample(groups):
    return groups
    samples = [[], []]
    samples[FIRST_GROUP] = random.sample(groups[FIRST_GROUP], SAMPLE_SIZE)
    samples[SECOND_GROUP] = random.sample(groups[SECOND_GROUP], SAMPLE_SIZE)
    return samples


def GetAGradient(sample, m, n):
    try:
       helper = m ** 2 + 1
    except:
       return 0,0

    derivative = 0
    sumLoss = 0
    for point in sample[FIRST_GROUP]:
        x0 = point[X_COORDINATE]
        y0 = point[Y_COORDINATE]
        loss = Distance(m, n, x0, y0, -1)
        if (loss > 0):
            sumLoss += loss
            derivative -= (x0 * (helper ** 0.5) - m * (m * x0 - y0 + n) / helper ** 0.5) / (helper)

    for point in sample[SECOND_GROUP]:
        x0 = point[X_COORDINATE]
        y0 = point[Y_COORDINATE]
        loss = Distance(m, n, x0, y0, 1)
        if(loss>0):
            sumLoss += loss
            derivative += (x0 * (helper ** 0.5) - m * (m * x0 - y0 + n) / helper ** 0.5) / (helper)

    return sumLoss, derivative




def GetBGradient(sample, m, n):
    derivative = 0
    sumLoss = 0

    try:
       helper = m ** 2 + 1
    except:
       return 0,0

    for point in sample[FIRST_GROUP]:
        x0 = point[X_COORDINATE]
        y0 = point[Y_COORDINATE]
        loss = Distance(m, n, x0, y0, -1)
        if (loss > 0):
            sumLoss += loss
            derivative -= 1 / (helper**0.5)

    for point in sample[SECOND_GROUP]:
        x0 = point[X_COORDINATE]
        y0 = point[Y_COORDINATE]
        loss = Distance(m, n, x0, y0, 1)
        if(loss>0):
            sumLoss += loss
            derivative += 1 / (helper**0.5)

    return sumLoss, derivative


# random initial step
a = random.uniform(-3,3)
b = random.uniform(-4,4)
xx = np.linspace(-4, 4, 2)
vectorHistory = []
vectorHistory.append([xx * a + b, "random"])
lossHistory = []
gradHistory = [[],[]]
aHistorty = []
bHistorty = []
print "start timing"
tt = -time.clock()
for i in range(NUM_ITERATIONS):
    sample = GetSample(groups)

    loss, grad = GetAGradient(sample, a, b)

    #update a
    if not grad == 0:
        newA = a - LEARNING_RATE * loss / grad
    else:
        newA = a

    if i % (NUM_ITERATIONS / HISTORY_TO_SHOW) == 0:  # Dont break matplot,dont show 1 million vectors
        gradHistory[0].append(grad)

    # update b
    loss, grad =GetBGradient(sample,newA,b)
    if not grad==0:
        newB = b - LEARNING_RATE * loss / grad
    else:
        newB=b

    if i % (NUM_ITERATIONS / HISTORY_TO_SHOW) == 0:  # Dont break matplot,dont show 1 million vectors
        aHistorty.append([a])
        bHistorty.append([b])
        lossHistory.append(loss)
        gradHistory[1].append(grad)

    prevA = a
    prevB = b
    a = newA
    b = newB

    ys = a * xx + b
    if (i % (NUM_ITERATIONS / VECTORS_TO_SHOW) == 0):
        vectorHistory.append([ys, "iteration" + str(i)])

tt += time.clock()
print "time passed: ", tt

# Vectors on graph
plt.ylim(-10, 10)
plt.xlabel("x")
plt.ylabel("y")
style.use("ggplot")
pointsPerColor = (len(vectorHistory)) / 8.0
colors = ["black", "gray", "red", "orange", "yellow", "cyan", "lime", "green", "darkgreen"]
for i in range(len(vectorHistory)):
    plt.plot(xx, vectorHistory[i][0], 'k-', label=vectorHistory[i][1], color=colors[int(i / pointsPerColor)])

# dots on graph
x1 = [point[X_COORDINATE] for point in groups[FIRST_GROUP]]
y1 = [point[Y_COORDINATE] for point in groups[FIRST_GROUP]]
x2 = [point[X_COORDINATE] for point in groups[SECOND_GROUP]]
y2 = [point[Y_COORDINATE] for point in groups[SECOND_GROUP]]

plt.scatter(np.array(x1), np.array(y1), c="blue")
plt.scatter(np.array(x2), np.array(y2), c="orange")

plt.legend()
plt.show()

print"last vector (" + str(aHistorty[-1][0]) + ")x" + " + " + str(bHistorty[-1][0])

pointsPerColor = (len(aHistorty)) / 8.0
for i in range(len(aHistorty)):
    plt.scatter(np.array([aHistorty[i]]), np.array([bHistorty[i]]), color=colors[int(i / pointsPerColor)])
plt.plot(aHistorty, bHistorty, 'k-', label="ab over time,black to green", color="purple")

plt.xlabel("a")
plt.ylabel("b")
plt.legend()
plt.show()

plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.plot([int(i * NUM_ITERATIONS / HISTORY_TO_SHOW) for i in range(len(lossHistory))], lossHistory, 'k-',
         label="loss @ iteration", color="blue")
plt.legend()
plt.show()

plt.xlabel("Iteration")
plt.ylabel("grad a")
plt.plot([int(i * NUM_ITERATIONS / HISTORY_TO_SHOW) for i in range(len(gradHistory[0]))], gradHistory[0], 'k-',
         label="grad @ iteration", color="purple")
plt.legend()
plt.show()

plt.xlabel("Iteration")
plt.ylabel("grad b")
plt.plot([int(i * NUM_ITERATIONS / HISTORY_TO_SHOW) for i in range(len(gradHistory[1]))], gradHistory[1], 'k-',
         label="grad @ iteration", color="purple")
plt.legend()
plt.show()



plt.xlabel("gradient")
plt.ylabel("loss")
plt.scatter(np.array(lossHistory), np.array(gradHistory[0]), c="a gradient")
plt.scatter(np.array(lossHistory), np.array(gradHistory[1]), c="b gradient")
plt.legend()
plt.show()
