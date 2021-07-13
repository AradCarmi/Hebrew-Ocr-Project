---------------------------------------------------------------------Recognizing Hebrew Text----------------------------------------------------------------------


Project overview

Phase 1- Detect the contours of the words in the image:
In the first part of the project we used the “East model” in order to detect words. In order to use the model, we had to construct a blob from the image and then forward it in the different layers of the model to obtain the two output sets. The output geometry map used to derive the bounding box coordinates of text in our input images. And similarly, the scores map, containing the probability of a given region containing text.


Phase 2 - Rectify and recognize the detected boxes:
In the second phase we used the pytesseract library. The input for the phase is the output from phase one which is the boxes. For each box which represent a word in the image we recognized the word using pytesseract function “image_to_string”. The function gets as input the boxes and config (we config the text to Hebrew) and print to the console the string that represent the word in the image. In order to use the function, we had to download the “tessdata” which is a pre trained model for Hebrew recognition.


How to run the project

•	Install the proper environment (Python, OpenCV, Tesseract, … ,etc.).
•	Download the EAST model from here. 
•	Download the dataset for the tesseract from here. 
•	Download the image to images folder that included in the project.
•	Run “python ./ Hebrew_Recogntion/ Recogntion_hebrew_in_wild.py [path to your image]”


