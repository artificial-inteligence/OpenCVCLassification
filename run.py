from tkinter import *
from tkinter.filedialog import askopenfilename
import cv2
from PIL import Image, ImageTk
import cv2.face
import glob
import random
import numpy as np

faceDet = cv2.CascadeClassifier("HAARFilters/haarcascade_frontalface_default.xml")
faceDet_two = cv2.CascadeClassifier("HAARFilters/haarcascade_frontalface_alt2.xml")
faceDet_three = cv2.CascadeClassifier("HAARFilters/haarcascade_frontalface_alt.xml")
faceDet_four = cv2.CascadeClassifier("HAARFilters/haarcascade_frontalface_alt_tree.xml")
emotions = ["anger", "disgust", "fear", "happy", "sadness", "surprise"]  # Emotion list
fishface = cv2.face.FisherFaceRecognizer_create()  # Initialize fisher face classifier
data = {}

class MainWindow:
    def __init__(self, master):
        self.normalPadding = 5
        self.photo = Image.open("working/prism.jpg")
        self.pickedPhoto = ImageTk.PhotoImage(self.photo)

        self.getImageBtn = Button(master, text="Get Image", bg="red", fg="white", padx=self.normalPadding, pady=self.normalPadding,command=self.selectImage)
        self.getImageBtn.grid(row=1, column=0)

        self.appTitleTxt = Label(master, text="Application Title", bg="green", fg="black", padx=self.normalPadding,pady=self.normalPadding)
        self.appTitleTxt.grid(row=0, columnspan=2, sticky=N + E + W)

        self.imgDisplayLbl = Label(master, image=self.pickedPhoto, width=350,height=350)
        self.imgDisplayLbl.grid(row=1,column=1, sticky=N+E+S+W)

        self.resultTxt = Label(root, text="Confidence: XY%", bg="green", fg="black", padx=self.normalPadding, pady=self.normalPadding,width=100)
        self.resultTxt.grid(columnspan=2, row=2, sticky=E + W)

    def selectImage(self):
        print("clicked")
        pickedFile = askopenfilename(initialdir="/", title="Select file",
                                     filetypes=(("jpg files", "*.jpg"), ("all files", "*.*")))

        original = cv2.imread(pickedFile)
        cv2.imwrite("working/selectedImg.jpg", original)
        self.preProcess("working/selectedImg.jpg", "working/preProcessed.jpg")

        self.photo = Image.open("working/preProcessed.jpg")
        self.pickedPhoto = ImageTk.PhotoImage(self.photo)
        self.imgDisplayLbl.configure(image=self.pickedPhoto)

        # Now run it
        pred ,conf = self.run_recognizer()

        # ============================
        self.resultTxt.config(text=str(pred) + " " + str(conf))


        # ============================
    def preProcess(self, inputImgPath, outputImgPath):
        toProcess = cv2.imread(inputImgPath)
        # filter
        bfil = cv2.bilateralFilter(toProcess, 9, 75, 75)
        # gray scale
        gray = cv2.cvtColor(bfil, cv2.COLOR_BGR2GRAY)
        # detect face
        # Detect face using 4 different classifiers
        face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                        flags=cv2.CASCADE_SCALE_IMAGE)
        face_two = faceDet_two.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                                flags=cv2.CASCADE_SCALE_IMAGE)
        face_three = faceDet_three.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                                    flags=cv2.CASCADE_SCALE_IMAGE)
        face_four = faceDet_four.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                                  flags=cv2.CASCADE_SCALE_IMAGE)
        # Go over detected faces, stop at first detected face, return empty if no face.
        if len(face) == 1:
            facefeatures = face
        elif len(face_two) == 1:
            facefeatures = face_two
        elif len(face_three) == 1:
            facefeatures = face_three
        elif len(face_four) == 1:
            facefeatures = face_four
        else:
            facefeatures = ""
        # focus on face
        for (x, y, w, h) in facefeatures:  # get coordinates and size of rectangle containing face
            gray = gray[y:y + h, x:x + w]  # Cut the frame to size
            try:
                out = cv2.resize(gray, (350, 350))  # Resize face so all images have same size
                cv2.imwrite(outputImgPath, out)  # Write image
            except:
                pass  # If error, pass file

    def get_files(self, emotion):  # Define function to get file list, randomly shuffle it and split 80/20
        files = glob.glob("dataset\\%s\\*" % emotion)
        training = files[:int(len(files) * 1)]
        return training

    def make_sets(self):
        training_data = []
        training_labels = []

        prediction_labels = []
        for emotion in emotions:
            training = self.get_files(emotion)
            # Append data to training and prediction list, and generate labels 0-7
            for item in training:
                image = cv2.imread(item)  # open image
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to gray scale
                training_data.append(gray)  # append image array to training data list
                training_labels.append(emotions.index(emotion))

                prediction_image = cv2.imread("working/preProcessed.jpg")
                prediction_filter = cv2.bilateralFilter(prediction_image, 9, 75, 75)
                prediction_gray = cv2.cvtColor(prediction_filter, cv2.COLOR_BGR2GRAY)

        return training_data, training_labels, prediction_gray

    def run_recognizer(self):
        training_data, training_labels, prediction_gray = self.make_sets()
        print("training fisher face classifier")
        print("size of training set is:", len(training_labels), "images")
        fishface.train(training_data, np.asarray(training_labels))
        print("predicting classification set")

        pred, conf = fishface.predict(prediction_gray)
        print("pred  " + str(pred))
        print("emotion inded   " + emotions[pred])
        print("conf   " + str(conf))
        prediction = emotions[pred]
        probability = conf
        return prediction, probability



root = Tk()
mainWin = MainWindow(root)
root.test = mainWin
root.minsize(600, 600)
root.maxsize(600, 600)


root.mainloop()
