import cv2
import cv2.face
import glob
import random
import numpy as np
emotions = ["anger", "disgust", "fear", "happy", "sadness", "surprise"]  # Emotion list
fishface = cv2.face.FisherFaceRecognizer_create()  # Initialize fisher face classifier
data = {}


def get_training_files(emotion):  # Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("dataset\\%s\\*" %emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)]  # get first 80% of file list
    prediction = files[-int(len(files)*0.2):]  # get last 20% of file list
    return training, prediction


def make_test_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        training, prediction = get_training_files(emotion)
        # Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item)  # open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to gray scale
            training_data.append(gray)  # append image array to training data list
            training_labels.append(emotions.index(emotion))
        for item in prediction:  # repeat above process for prediction set
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prediction_data.append(gray)
            prediction_labels.append(emotions.index(emotion))
    return training_data, training_labels, prediction_data, prediction_labels


def run_test_recognizer():
    training_data, training_labels, prediction_data, prediction_labels = make_test_sets()
    print("training fisher face classifier")
    print("size of training set is:", len(training_labels), "images")
    fishface.train(training_data, np.asarray(training_labels))
    print("predicting classification set")
    cnt = 0
    correct = 0
    incorrect = 0
    sum_conf = 0
    for image in prediction_data:
        pred, conf = fishface.predict(image)
        if pred == prediction_labels[cnt]:
            correct += 1
            cnt += 1
            sum_conf += conf
            print("pred  " + str(pred))
            print("emotion    " + emotions[pred])
            print("conf   " + str(conf))


        else:
            incorrect += 1
            sum_conf += conf
            cnt += 1
    print("average conf = " + str(sum_conf) + "/" + str(cnt) + " = " + str((sum_conf/cnt)))

    return ((100*correct)/(correct + incorrect),(sum_conf/cnt))
# Now run it
metascore = []
distanceMeta = []
for i in range(0,10):
    correct, avgDistance = run_test_recognizer()
    print("got", correct, "percent correct!")
    metascore.append(correct)
    distanceMeta.append(avgDistance)
print("\n\nend score:", np.mean(metascore), "percent correct!")
print("\n\nend score (Distance):", np.mean(distanceMeta), "average Distance for above percentage")
