from flask import Blueprint, render_template, Flask, flash, request, redirect, url_for, redirect
from flask_login import login_required, current_user
import os
from werkzeug.utils import secure_filename
from src import create_app

import cv2
import numpy as np
from matplotlib import pyplot as plt

views = Blueprint("views", __name__)
app = create_app()


@views.route("/sign-up")
def signUp():
    return render_template("signUp.html", user=current_user)


@views.route("/")
@login_required
def home():
    return render_template("home.html", user=current_user)


app.config["IMAGE_UPLOADS"] = "src/static/image_upload"
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["PNG", "JPG", "JPEG", ]
folder_images = os.listdir("src/static/image_upload")


def allowed_image(filename):
    if not "." in filename:
        return False
    ext = filename.rsplit(".", (1))[1]
    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False


@views.route("/image_upload", methods=["GET", "POST"])
def image_upload():
    if request.method == "POST":

        if request.files:
            image = request.files["image"]
            print("Image Saved!")

            if image.filename == "":
                print("Image Must Have A Filename")
                return redirect(request.url)

            if not allowed_image(image.filename):
                print("That image extension in not allowed!")
                return redirect(request.url)
            else:
                filename = secure_filename(image.filename)
                filename = "image.jpeg"
            image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))
            return redirect(url_for("views.specify"))
    return render_template("image_upload.html", user=current_user)


@views.route("/specify")
def specify():
    net = cv2.dnn.readNet('assets/yolov3.weights', 'assets/yolov3.cfg')
    classes = []
    # Passing the names of the txt file in a array
    with open('assets/coco_names.txt', 'r') as f:
        classes = f.read().splitlines()

    # Storing the image and take the with, the height and the depth
    img = plt.imread("src/static/image_upload/image.jpeg")
    height, width, deapth = img.shape

    # Make sure all the imported image will have the same size and also scalling down the image
    blob = cv2.dnn.blobFromImage(
    img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
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

    fig, axes = plt.subplots()

    for i in boxes.flatten():
        x, y, w, h = pounding_boxes[i]
        label = str(classes[class_ids[i]])
        prediction = str(round(predictions[i], 2) * 100)
        color = colors[i]

        rect = plt.Rectangle((x, y), w, h, linewidth=1,
                         edgecolor='b', facecolor='none')
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
            quantity_labels.append(
                len([temp_label for temp_label in all_labels if value == temp_label]))

        axes.imshow(img)
        plt.savefig("src/static/predicted_image.jpeg", bbox_inches="tight")

    return render_template("specify.html", user=current_user)
