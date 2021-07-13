import sys
import numpy as np
import cv2
import pytesseract
from imutils.object_detection import non_max_suppression

def detect(args):

    image = cv2.imread(args['image'])
    # Saving a original image and shape
    orig = image.copy()
    (origH, origW) = image.shape[:2]

    # set the new height and width to default 320 by using args #dictionary.
    (newW, newH) = (args["width"], args["height"])

    # Calculate the ratio between original and new image for both height and weight.
    # This ratio will be used to translate bounding box location on the original image.
    rW = origW / float(newW)
    rH = origH / float(newH)

    # resize the original image to new dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # construct a blob from the image to forward pass it to EAST model
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)

    # load the pre-trained EAST model for text detection
    net = cv2.dnn.readNet(args["east"])

    # The following two layer need to pulled from EAST model for achieving this.
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    # Forward pass the blob from the image to get the desired output layers
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    # call the private function predictions
    (rects, confidence_val) = predictions(scores,geometry)

    boxes = non_max_suppression(np.array(rects), probs=confidence_val)

    img2 = orig.copy()
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        # draw the bounding box on the image
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # show the output image
    cv2.imshow("Text Detection", orig)
    cv2.waitKey(0)

    # call the private function recognition
    recognition(boxes, img2, rW, rH)

# Returns a bounding box and probability score if it is more than minimum confidence
def predictions(prob_score, geo):
    (numR, numC) = prob_score.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, numR):
        scoresData = prob_score[0, 0, y]
        x0 = geo[0, 0, y]
        x1 = geo[0, 1, y]
        x2 = geo[0, 2, y]
        x3 = geo[0, 3, y]
        anglesData = geo[0, 4, y]

        for i in range(0, numC):
            if scoresData[i] < args["min_confidence"]:
                continue

            (offX, offY) = (i * 4.0, y * 4.0)

            # extracting the rotation angle for the prediction and computing the sine and cosine
            angle = anglesData[i]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # using the geo volume to get the dimensions of the bounding box
            h = x0[i] + x2[i]
            w = x1[i] + x3[i]

            # compute start and end for the text pred bbox
            endX = int(offX + (cos * x1[i]) + (sin * x2[i]))
            endY = int(offY - (sin * x1[i]) + (cos * x2[i]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[i])

    # return bounding boxes and associated confidence_val
    return (rects, confidences)


def recognition(boxes, orig, rW, rH):
    (origH, origW) = orig.shape[:2]

    # initialize the list of results
    results = []
    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        # in order to obtain a better OCR of the text we can potentially
        # apply a bit of padding surrounding the bounding box
        dX = int((endX - startX) * 0.05)

        dY = int((endY - startY) * 0.1)

        # apply padding to each side of the bounding box, respectively
        startX = max(0, startX - dX)
        startY = max(0, startY - dY)
        endX = min(origW, endX + (dX * 2))
        endY = min(origH, endY + (dY * 2))

        roi = orig[startY:endY, startX:endX]
        # config - -l : hebrew, --psm: 8 for one word recognition
        config = ("-l heb --oem 1 --psm 8")
        text = pytesseract.image_to_string(roi, config=config)
        # add the bounding box coordinates and OCR'd text to the list of results
        results.append(((startX, startY, endX, endY), text))

    # sort the results bounding box
    # coordinates from top to bottom
    results = sorted(results, key=lambda r: r[0][1])
    # loop over the results
    for ((startX, startY, endX, endY), text) in results:
        print("========")
        print("{}\n".format(text))

        # using OpenCV, then draw the text and a bounding box surrounding
        # the text region of the input image
        text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
        output = orig.copy()
        cv2.rectangle(output, (startX, startY), (endX, endY),
                      (0, 0, 255), 2)
        # show the output image
        cv2.imshow("Text Recognition", output)
        cv2.waitKey(0)

if __name__ == '__main__':
    # Creating argument dictionary for the default arguments needed in the code.
    args = {"image": sys.argv[1], "east": "../frozen_east_text_detection.pb", "min_confidence": 0.5, "width": 320,
            "height": 320}

    detect(args)



