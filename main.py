import cv2
import numpy as np
from matplotlib import pyplot as plt


net = cv2.dnn.readNet('assets/yolov3.weights', 'assets/yolov3.cfg')
classes = []

# Passing the names of the txt file in a array
with open('assets/coco_names.txt', 'r') as f:
    classes = f.read().splitlines()

# Storing the image and take the with, the height and the depth
img = plt.imread('assets/image.jpeg')
height, width, deapth = img.shape

# Make sure all the imported image will have the same size and also scalling down the image
blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
net.setInput(blob)
output_layers = net.getUnconnectedOutLayersNames()
layerOutputs = net.forward(output_layers)

pounding_boxes = []
predictions = []
class_ids = []

for output in layerOutputs:
    for detection in output:

        # Stores all the classes predictions of the image, finding the location of the highest score and then store it
        scores = detection[5:]
        class_id = np.argmax(scores)
        prediction = scores[class_id]

        if prediction > 0.5:

            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w/2)
            y = int(center_y - h/2)

            # Finding out how many objects image contains
            pounding_boxes.append([x, y, w, h])
            predictions.append((float(prediction)))
            class_ids.append(class_id)

# Finding out how many objects found
boxes = cv2.dnn.NMSBoxes(pounding_boxes, predictions, 0.5, 0.4)
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(pounding_boxes), 3))

# Showing in the new image the pounding boxes with label name and a prediction

distinct_labels = []
quantity_labels = []
all_labels = []


bar_fig, bar_axes = plt.subplots()
fig, axes = plt.subplots()

for i in boxes.flatten():
    x, y, w, h = pounding_boxes[i]
    label = str(classes[class_ids[i]])
    prediction = str(round(predictions[i], 2) * 100)
    color = colors[i]

    rect = plt.Rectangle((x, y), w, h, linewidth = 1, edgecolor = 'b', facecolor = 'none')
    label_text = plt.text((x + 2), (y + 20), label)
    prediction_text = plt.text((x + 2), (y + 37), prediction + "%")
    plt.xticks([])
    plt.yticks([])
    axes.add_patch(rect)

    distinct_labels.append(label)
    all_labels.append(label)
    distinct_labels = list(set(distinct_labels))

    quantity_labels = []
    for value in distinct_labels:
        quantity_labels.append(len([temp_label for temp_label in all_labels if value == temp_label]))

# Making a bar chart that shows the quantity of every object in the image
bar_chart = bar_axes.bar(distinct_labels, quantity_labels,width = 0.4 ,color='blue')
axes.imshow(img)
plt.show()