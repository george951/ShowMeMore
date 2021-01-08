import cv2
import numpy as np

net = cv2.dnn.readNet('assets/yolov3.weights','assets/yolov3.cfg')
classes = []

# Passing the names of the txt file in a array
with open('assets/coco_names.txt','r') as f:
    classes = f.read().splitlines()

# Storing the image and take the with, the height and the depth 
img = cv2.imread('assets/image.jpeg')
height, width, deapth = img.shape

# Make sure all the imported image will have the same size and also scalling down the image
blob = cv2.dnn.blobFromImage(img, 1/255, (416,416), (0,0,0), swapRB=True, crop=False)
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

#Showing in the new image the pounding boxes with label name and a prediction 
for i in boxes.flatten():
    x, y, w, h = pounding_boxes[i]
    label = str(classes[class_ids[i]])
    prediction = str(round(predictions[i], 2))
    color = colors[i]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(img, label + " " + prediction, (x, y + 20), font, 2, (255,255,255), 2)


cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
