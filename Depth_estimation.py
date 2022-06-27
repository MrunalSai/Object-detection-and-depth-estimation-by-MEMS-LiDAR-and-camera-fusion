import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import ransac, LineModelND
img = cv.imread("C:/scientific project/sample - 12/023153.jpg")

confThreshold = 0.5
nmsThreshold = 0.2

#### LOAD MODEL
## Coco Names
classesFile = "coco.names"
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

## Model Files
modelConfiguration = "yolov3-spp.cfg"
modelWeights = "yolov3-spp.weights"
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

list1 = []
def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]

            classId = np.argmax(scores)
            confidence = scores[classId]

            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    for i in indices:

        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]

        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
                   (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        list1.append([x,y,w,h])




blob = cv.dnn.blobFromImage(img, 1 / 255.0, (416, 416), [0, 0, 0], 1, crop=False)

net.setInput(blob)
layersNames = net.getLayerNames()

outputNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()]

outputs = net.forward(outputNames)

findObjects(outputs, img)


a = np.loadtxt("C:/scientific project/sample - 12/points.txt")
print(f'shape of mems_lidar points is {a.shape}')
print(f'info of bounding boxes {list1}')

point_list = []

for index,value in enumerate(list1):

    list2 = []
    for x in a:

        if (x[0] >= list1[index][0] and x[0] <= (list1[index][0] + list1[index][2])) and (x[1] >= list1[index][1] and x[1] <= (list1[index][1] + list1[index][3])):
            list2.append(x[2])

    point_list.append(list2)


#Implementation of ransac

def rns(point_list):
    list4 = []
    list5 = []
    y1 = np.array(point_list)
    x1 = np.linspace(0,100,len(point_list))

    data = np.column_stack([x1,y1])

    nd_model = LineModelND()
    nd_model.estimate(data)
    nd_model.params
    origin, direction = nd_model.params

    plt.plot(x1,y1,'.')
    l1 = np.arange(0,100)

    plt.plot(l1, nd_model.predict_y(l1), 'r-')

    model_robust, inliers = ransac(data, LineModelND, min_samples=2, residual_threshold=1, max_trials=1500)

    outliers = (inliers == False)
    yy = model_robust.predict_y(l1)
    fig, ax = plt.subplots()
    plt.xlabel("Points")
    plt.ylabel("Depth")
    plt.title("RANSAC")
    ax.plot(data[inliers,0], data[inliers,1], '.r', alpha=0.6, label = 'inlier data')
    ax.plot(data[outliers,0], data[outliers,1], '.b', alpha=0.6, label = 'outlier data')
    ax.plot(l1, yy, '-g', label = 'Robust Line Model' )
    ax.legend(loc = "upper right")
    plt.show()

    for i,item in enumerate(inliers):
        if item == True:
            list4.append(i)


    for i,item in enumerate(list4):
        for a,value in enumerate(point_list):
            if a == item:
                list5.append(value)

    max1 = max(list5)
    return max1

for i,value in enumerate(point_list):

    print(f'depth of obj {i+1} is {rns(point_list[i])}')

cv.imshow('Image', img)
cv.waitKey(0)
cv.destroyAllWindows()