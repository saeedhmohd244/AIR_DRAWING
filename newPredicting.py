import os
import numpy as np
import cv2
import csv
import glob
import pandas as pd
from keras.models import load_model
# from detection import CharacterDetector

# det = CharacterDetector(loadFile="model_hand.h5")
img = cv2.imread('new.jpg')


class CharsPredicting:

    characters = []

    def process_characters_folder():

        characters_folder = "characters/0"
        if not os.path.exists(characters_folder):
            print("Error: 'characters' folder not found.")
            return
        
        
        CharsPredicting.characters.clear()
        new_string = ''

        # for filename in os.listdir(characters_folder):
        #     if filename.endswith(".png"):
        #         filepath = os.path.join(characters_folder, filename)
        #         print(f"Processing {filepath}")
        #         prediction = det.predict(filepath)
        #         # os.remove(filepath)
        #         CharsPredicting.characters.append(prediction)
        #         print(prediction)

        header  =["label"]
        for i in range(0,784):
            header.append("pixel"+str(i))
        with open('new.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    
        for label in range(1):
#    dirList = glob.glob("pre/"+str(label)+"/*.jpg")
            dirList = glob.glob("characters/"+str(label)+"/*.png") 
            for img_path in dirList:
                im= cv2.imread(img_path)
                os.remove(img_path)
                im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
                im_gray = cv2.GaussianBlur(im_gray,(15,15), 0)
                roi= cv2.resize(im_gray,(28,28), interpolation=cv2.INTER_AREA)
        
                data=[]
                data.append(label)
                rows, cols = roi.shape
        
       ## Fill the data array with pixels one by one.
                for i in range(rows):
                    for j in range(cols):
                        k =roi[i,j]
                        if k>100:
                            k=1
                        else:
                            k=0
                        data.append(k)
                with open('new.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(data)






        new_df = pd.read_csv('new.csv')

# Preprocess the new data
        new_data = new_df.iloc[:, 1:].values.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        model = load_model('modelalphadigit25.h5')

# Make predictions
        predictions = model.predict(new_data)

# Get the predicted digit (class with the highest probability)
        predicted_labels = np.argmax(predictions, axis=1)

        print("Predicted Labels:", predicted_labels)


        labels = predicted_labels

        classes = {
        "0": "0",
        "1": "1",
        "2": "2",
        "3": "3",
        "4": "4",
        "5": "5",
        "6": "6",
        "7": "7",
        "8": "8",
        "9": "9",
        "10": "A",
        "11": "B",
        "12": 'C',
        "13": "D",
        "14":  "E",
        "15": "F",
        "16": 'G',
        "17": "H",
        "18": "I",
        "19": "J",
        "20": "K",
        "21":"L",
        "22":"M",
        "23":"N",
        "24":"O",
        "25":"P",
        "26":"Q",
        "27":"R",
        "28":"S",
        "29":"T",
        "30":"U",
        "31":"v",
        "32":"W",
        "33":"X",
        "34":"Y",
        "35":"Z",
        "36":"a",
        "37":"b",
        "38":"c",
        "39":"d",
        "40":"e",
        "41":"f",
        "42":"g",
        "43":"h",
        "44":"i",
        "45":"j",
        "46":"k",
        "47":"l",
        "48":"m",
        "49":"n",
        "50":"o",
        "51":"p",
        "52":"q",
        "53":"r",
        "54":"s",
        "55":"t",
        "56":"U",
        "57":"v",
        "58":"w",
        "59":"x",
        "60":"y",
        "61":"z",
        }
  

        characters = [classes[str(c)] for c in labels]

        print(characters)

        words = ','.join([str(item) for item in characters])

        symbol_to_remove = ","

        new_string = ""
        for char in words:
            if char != symbol_to_remove:
                new_string += char


        print(new_string)
        print(words)


        # print("Predicted characters:", CharsPredicting.characters)

        # words = ''.join(CharsPredicting.characters)

        print("Predicted words:", words)

        os.remove('new.csv')
        
        return(new_string)
        