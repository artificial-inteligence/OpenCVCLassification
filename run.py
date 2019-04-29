import os
import random
from tkinter import *
from tkinter.filedialog import askopenfilename
import cv2
from PIL import Image, ImageTk
import cv2.face
import glob
import numpy as np

# HAAR filters used for face feature detection
faceDet = cv2.CascadeClassifier("HAARFilters/haarcascade_frontalface_default.xml")
faceDet_two = cv2.CascadeClassifier("HAARFilters/haarcascade_frontalface_alt2.xml")
faceDet_three = cv2.CascadeClassifier("HAARFilters/haarcascade_frontalface_alt.xml")
faceDet_four = cv2.CascadeClassifier("HAARFilters/haarcascade_frontalface_alt_tree.xml")
emotions = ["anger", "disgust", "fear", "happy", "sadness", "surprise"]  # Emotion list
# number of components  0 is best, the default.   threashold = 1000 how big a distance you can have between results
# and predictions before it returns -1 (fail) setting it to 1000 gives us 0 =100% confidence 1000 = 0% confidence.
# need to run labeled tests to figure out what a reasonable threashold is for our dataset....
# accuracy can be predicted from there
# Labeled Tests Return values

# 903 highest distance with shuffled sets

# 61.96%    == 530 Distance      1% = 530/61.96 = 8.6   0% = 8.6*100 = 869
# 58.4%     == 501 Distance      1% = 501/58.4  = 8.6
fishFaceThreshold = 869
fishface = cv2.face.FisherFaceRecognizer_create(0, fishFaceThreshold)  # Initialize fisher face classifier
defaultImageLocation = "working/logo.jpg"
selectedImageLocation = "working/selectedImg.jpg"
preProcessedImageLocation = "working/preProcessed.jpg"
applicationTitle = "Emotion Recognition Helper"
trainedModel = "working/trainedFisherFace.yml"

# the  class that makes up th User interface
class MainWindow:
    def __init__(self, master):
        self.metascore = []
        self.distanceMeta = []
        self.normalPadding = 5
        self.photo = Image.open(defaultImageLocation)
        self.pickedPhoto = ImageTk.PhotoImage(self.photo)

        self.getImageBtn = Button(master, text="Get Image", bg="red", fg="white", padx=self.normalPadding, pady=self.normalPadding,command=self.selectImage)
        self.getImageBtn.grid(row=1, column=0, sticky=E)
        self.getImageBtn = Button(master, text="Test Me", bg="blue", fg="white", padx=self.normalPadding, pady=self.normalPadding, command=self.testAccuracy)
        self.getImageBtn.grid(row=1, column=0, sticky=W, padx=10)

        self.getImageBtn = Button(master, text="Update Training", bg="green", fg="white", padx=self.normalPadding, pady=self.normalPadding, command=self.saveTraining)
        self.getImageBtn.grid(row=1, column=0, padx=10)

        self.appTitleTxt = Label(master, text=applicationTitle, bg="green", fg="black", padx=self.normalPadding,pady=self.normalPadding)
        self.appTitleTxt.grid(row=0, columnspan=2, sticky=N + E + W)

        self.imgDisplayLbl = Label(master, image=self.pickedPhoto, width=350, height=350)
        self.imgDisplayLbl.grid(row=1,column=1, sticky=N+E+S+W)

        self.resultTxt = Label(root, text="Confidence: XY%", bg="green", fg="black", padx=self.normalPadding, pady=self.normalPadding,width=100)
        self.resultTxt.grid(columnspan=2, row=2, sticky=E + W)
    #
    def make_test_sets(self):
        # read and label the testing images for model testing
        prediction_data = []
        prediction_labels = []
        for emotion in emotions:
            prediction = self.get_training_files(emotion)
            for item in prediction:  # repeat above process for prediction set
                image = cv2.imread(item)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                prediction_data.append(gray)
                prediction_labels.append(emotions.index(emotion))

        return prediction_data, prediction_labels

    def run_test_recognizer(self):
        # run the recognizer against the training images to return accuracy
        prediction_data, prediction_labels = self.make_test_sets()
        print("size of training set is:", len(prediction_labels), "images")
        # get trined model to test or create one if not found
        if os.path.isfile(trainedModel):
            fishface.read(trainedModel)
        else:
            print(".yml file not found. Training the FisherFace classifier")
            self.saveTraining()
            print("Saving new Training yml file")
            fishface.read(trainedModel)
        cnt = 0
        correct = 0
        incorrect = 0
        total_distance = 0
        highest_pred = 0
        # iterate through predictions and tally results
        for raw_image in prediction_data:
            cv2.imwrite("working/training_RawImg.jpg", raw_image)
            self.preProcess("working/training_RawImg.jpg", "working/training_ProcessedImg.jpg")
            image = cv2.imread("working/training_ProcessedImg.jpg")
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            pred, dist = fishface.predict(gray_image)
            if dist > highest_pred:
                highest_pred = dist
                print(str(highest_pred)   + " :highest Distance")
            if pred == prediction_labels[cnt]:
                correct += 1
                cnt += 1
                print(dist, " distance added to total")
                total_distance += dist
            else:
                incorrect += 1
                cnt += 1
                print(dist, " distance added to total")
                total_distance += dist

            print(total_distance, " : new total distance")
        avg_dist = total_distance/cnt
        return ((100 * correct) / (correct + incorrect)), avg_dist

    def get_training_files(self, emotion):
        # Define function to get file list, randomly shuffle it
        files = glob.glob("dataset_testing\\%s\\*" % emotion)
        random.shuffle(files)
        prediction = files[:int(len(files) * 1)]
        return prediction

    def testAccuracy(self):
        # onClick function: test the systems accuracy against the test dataSet
        correct, avg_dist = self.run_test_recognizer()
        print("got", correct, "percent correct!")
        print(avg_dist, " average distance")
        self.metascore.append(correct)
        self.distanceMeta.append(avg_dist)
        print("\n\nend score:", np.mean(self.metascore), "percent correct!")
        print("\n\nend score:", np.mean(self.distanceMeta), "average distance!")
        self.resultTxt.config(text=str(np.mean(self.metascore)) + " percent correct!", bg="gold")

    def selectImage(self):
        # select image
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
        # prepair the image
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
        pred, distance = self.run_recognizer()

        # ============================
        print(distance)
        # 0 distance == 100% (perfect match with trained recognizers idea of mood)
        # distance = fisherFaceThreshold = 0% (I'ts as far away from the trained models idea of a mood as possible)

        # ===
        # could not identify mood
        if pred == -1:
            # distance is maximum as we have no idea
            distance = fishFaceThreshold
        if distance > fishFaceThreshold:
            distance = fishFaceThreshold
        # flip it the normal way around for calculating averages:
        # so if dist = 0 distance = fisherFaceThreshold (100%)
        # # , if distance  = fisherFaceThreshold distance = 0 (0%)
        distance = fishFaceThreshold - distance

        # ===
        test = distance / fishFaceThreshold
        percentage = round((test * 100), 2)
        self.resultTxt.config(text=str(pred) + " " + str(percentage) + "%", bg="green")

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

        face_one_one = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(5, 5),
                                        flags=cv2.CASCADE_SCALE_IMAGE)
        face_two_one = faceDet_two.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(5, 5),
                                                flags=cv2.CASCADE_SCALE_IMAGE)
        face_three_one = faceDet_three.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(5, 5),
                                                    flags=cv2.CASCADE_SCALE_IMAGE)
        face_four_one = faceDet_four.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(5, 5),
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
            print("face Found using final")
            # running filters again but on smaller scale
        elif len(face_four) == 1:
            facefeatures = face_one_one
            print("face Found using 1st filter small min neighbours")
        elif len(face_four) == 1:
            facefeatures = face_two_one
            print("face Found using 2nd filter small min neighbours")
        elif len(face_four) == 1:
            facefeatures = face_three_one
            print("face Found using 3rd filter small min neighbours")
        elif len(face_four) == 1:
            facefeatures = face_four_one
            print("face Found using final filter small min neighbours")

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
        # get training files form disk
        files = glob.glob("dataset\\%s\\*" % emotion)
        training = files[:int(len(files) * 1)]
        return training

    def run_recognizer(self):
        # run the recognizer on the processed image
        prediction_image = cv2.imread(preProcessedImageLocation)
        prediction_gray = cv2.cvtColor(prediction_image, cv2.COLOR_BGR2GRAY)
        if os.path.isfile(trainedModel):
            fishface.read(trainedModel)
        else:
            print("yml file not found. Training the FisherFace classifier")
            self.saveTraining()
            print("Saving new Training yml file")
            fishface.read(trainedModel)

        print("predicting classification..")
        pred, conf = fishface.predict(prediction_gray)
        print("pred  " + str(pred))
        print("emotion identified   " + emotions[pred])
        print("conf   " + str(conf))
        prediction = emotions[pred]
        probability = conf
        return prediction, probability

    def saveTraining(self):
        # update the trained model to the new DataSet (assuming files were manually added to it
        training_data = []
        training_labels = []
        # update the model
        for emotion in emotions:
            training = self.get_files(emotion)
            # Append data to training and prediction list with labels
            for item in training:
                image = cv2.imread(item)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                training_data.append(gray)
                training_labels.append(emotions.index(emotion))
        fishface.train(training_data, np.asarray(training_labels))
        fishface.save(trainedModel)


root = Tk()
mainWin = MainWindow(root)
root.test = mainWin
root.minsize(600, 600)
root.maxsize(600, 600)


root.mainloop()
