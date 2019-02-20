from tkinter import *
from tkinter.filedialog import askopenfilename
import cv2
from PIL import Image, ImageTk
import cv2.face
import glob
import numpy as np

faceDet = cv2.CascadeClassifier("HAARFilters/haarcascade_frontalface_default.xml")
faceDet_two = cv2.CascadeClassifier("HAARFilters/haarcascade_frontalface_alt2.xml")
faceDet_three = cv2.CascadeClassifier("HAARFilters/haarcascade_frontalface_alt.xml")
faceDet_four = cv2.CascadeClassifier("HAARFilters/haarcascade_frontalface_alt_tree.xml")
emotions = ["anger", "disgust", "fear", "happy", "sadness", "surprise"]  # Emotion list
# number of components  0 is best, the default.   threashold = 1000 how big a distance you can have between results
# and predictions before it returns -1 (fail) setting it to 1000 gives us 0 =100% confidence X = 0% confidence.
# accuracy can be predicted from there
fishface = cv2.face.FisherFaceRecognizer_create(0, 1300)  # Initialize fisher face classifier
defaultImageLocation = "working/logo.jpg"
selectedImageLocation = "working/selectedImg.jpg"
preProcessedImageLocation = "working/preProcessed.jpg"
applicationTitle = "Emotion Recognition Helper"

class MainWindow:
    def __init__(self, master):
        self.normalPadding = 5
        self.photo = Image.open(defaultImageLocation)
        self.pickedPhoto = ImageTk.PhotoImage(self.photo)

        self.getImageBtn = Button(master, text="Get Image", bg="red", fg="white", padx=self.normalPadding, pady=self.normalPadding,command=self.selectImage)
        self.getImageBtn.grid(row=1, column=0)

        self.appTitleTxt = Label(master, text=applicationTitle, bg="green", fg="black", padx=self.normalPadding,pady=self.normalPadding)
        self.appTitleTxt.grid(row=0, columnspan=2, sticky=N + E + W)

        self.imgDisplayLbl = Label(master, image=self.pickedPhoto, width=350,height=350)
        self.imgDisplayLbl.grid(row=1,column=1, sticky=N+E+S+W)

        self.resultTxt = Label(root, text="Confidence: XY%", bg="green", fg="black", padx=self.normalPadding, pady=self.normalPadding,width=100)
        self.resultTxt.grid(columnspan=2, row=2, sticky=E + W)

    def selectImage(self):
        print("clicked")
        pickedFile = askopenfilename(initialdir="/", title="Select file",
                                     filetypes=(("jpg files", "*.jpg"), ("all files", "*.*")))
        if pickedFile == "":
            self.resultTxt.config(text="Image Not Selected", bg="red")
            self.updateImage(defaultImageLocation)
            return
        original = cv2.imread(pickedFile)
        writeSuccess = cv2.imwrite(selectedImageLocation, original)
        if writeSuccess is False:
            self.resultTxt.config(text="Could Not Write Selected Image.", bg="red")
            self.updateImage(defaultImageLocation)
            return

        processResult = self.preProcess(selectedImageLocation, preProcessedImageLocation)
        if processResult == "noFace":
            self.resultTxt.config(text="Could Not Process Image. No Face Found", bg="red")
            self.updateImage(defaultImageLocation)
            return
        if processResult == "writeProcessedFail":
            self.resultTxt.config(text="Could Not Save Processed Image", bg="red")
            self.updateImage(defaultImageLocation)
            return

        self.updateImage(preProcessedImageLocation)

        # Now run it
        pred, conf = self.run_recognizer()

        # ============================
        self.resultTxt.config(text=str(pred) + " " + str(conf), bg="green")

        # ============================

    def updateImage(self,imageLocation):
        self.photo = Image.open(imageLocation)
        self.pickedPhoto = ImageTk.PhotoImage(self.photo)
        self.imgDisplayLbl.configure(image=self.pickedPhoto)

    def preProcess(self, inputImgPath, outputImgPath):
        toProcess = cv2.imread(inputImgPath)
        # filter
        bfil = cv2.bilateralFilter(toProcess, 9, 75, 75)
        # gray scale
        gray = cv2.cvtColor(bfil, cv2.COLOR_BGR2GRAY)
        # detect face
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
            print("face Found using 1st filter")
        elif len(face_two) == 1:
            facefeatures = face_two
            print("face Found using 2nd filter")
        elif len(face_three) == 1:
            facefeatures = face_three
            print("face Found using 3rd filter")
        elif len(face_four) == 1:
            facefeatures = face_four
            print("face Found using final filter")
        else:
            facefeatures = ""
            print("Could Not Find Face in Image")
            self.resultTxt.config(text="No Face Found In Image")
            return "noFace"
        # focus on face
        for (x, y, w, h) in facefeatures:  # get coordinates and size of rectangle containing face
            gray = gray[y:y + h, x:x + w]  # Cut the frame to size
            try:
                out = cv2.resize(gray, (350, 350))  # Resize face so all images have same size
                preProcessSaved = cv2.imwrite(outputImgPath, out)  # Write image
                if preProcessSaved is False:
                    self.resultTxt.config(text="Could Not Write Processed Image.")
                    return "writeProcessedFail"
            except:
                pass  # If error, pass file
                print("Error, Passing File")

    def get_files(self, emotion):
        files = glob.glob("dataset\\%s\\*" % emotion)
        training = files[:int(len(files) * 1)]
        return training

    def make_sets(self):
        training_data = []
        training_labels = []

        for emotion in emotions:
            training = self.get_files(emotion)
            # Append data to training and prediction list with labels
            for item in training:
                image = cv2.imread(item)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                training_data.append(gray)
                training_labels.append(emotions.index(emotion))

                prediction_image = cv2.imread(preProcessedImageLocation)
                prediction_gray = cv2.cvtColor(prediction_image, cv2.COLOR_BGR2GRAY)

        return training_data, training_labels, prediction_gray

    def run_recognizer(self):
        training_data, training_labels, prediction_gray = self.make_sets()
        print("training fisher face classifier")
        print("size of training set is:", len(training_labels), "images")
        fishface.train(training_data, np.asarray(training_labels))
        print("predicting classification set")

        pred, conf = fishface.predict(prediction_gray)
        print("pred  " + str(pred))
        print("emotion identified   " + emotions[pred])
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
