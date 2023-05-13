import cv2
import enum
import torch
import time
import numpy as np
import imageio as iio
import RPi.GPIO as GPIO

from transformers import AutoFeatureExtractor, AutoModelForImageClassification

try:
    extractor = AutoFeatureExtractor.from_pretrained("yangy50/garbage-classification")
    model = AutoModelForImageClassification.from_pretrained("yangy50/garbage-classification")
except:
    time.sleep(2)
    extractor = AutoFeatureExtractor.from_pretrained("yangy50/garbage-classification")
    model = AutoModelForImageClassification.from_pretrained("yangy50/garbage-classification")

class TrashEnum(enum.IntEnum):
    CARDBOARD = 0
    GLASS = 1
    METAL = 2
    PAPER = 3
    PLASTIC = 4
    TRASH = 5
    WAIT = 6


id_aggregation = {
    TrashEnum.CARDBOARD: TrashEnum.PAPER,  # cardboard -> paper
    TrashEnum.GLASS: TrashEnum.GLASS,  # glass -> glass
    TrashEnum.METAL: TrashEnum.PLASTIC,  # metal -> plastic
    TrashEnum.PAPER: TrashEnum.PAPER,  # paper -> paper
    TrashEnum.PLASTIC: TrashEnum.PLASTIC,  # plastic -> plastic
    TrashEnum.TRASH: TrashEnum.TRASH,   # trash -> trash
    TrashEnum.WAIT: TrashEnum.WAIT 
}

id2label = { #to zwraca tylko 3,4,5 i 1 - to dobrze przekodowac zeby bylo klarowniejsze
    TrashEnum.PAPER: 'paper',
    TrashEnum.GLASS: 'glass',
    TrashEnum.PLASTIC: 'plastic/metal',
    TrashEnum.TRASH: 'trash',
    TrashEnum.WAIT: "wait" #wait mowi o tym ze jak prawdopodobienstwo jest ponizej 70 to ma czekac (inna dioda sie pali)
}


def process_image(picture): #function of prediction
      # File to array
    inputs = extractor(picture, return_tensors="pt")  # Feature extraction from the image for model
    outputs = model(**inputs)  # Model inference: prediction
    pred = outputs.logits.softmax(1)  # Final prediction
    label_num = pred.argmax(1)  # The predicted class is "argmax" - index with maximal value
    label_num = label_num.item()
    pred_class_id = id_aggregation[label_num]  # paper/glass/plastic/trash id
    pred_class_name = id2label[pred_class_id]  # paper/glass/plastic/trash label
    probability = round(pred[:, pred.argmax(1)].flatten().item() * 100) #round for percentage value -> probability system 
    #.header(f'This image is {pred_class_name}'
               # f' with probability: {probability}%')
    final_class = pred_class_id if probability > 70 else TrashEnum.WAIT #if porbability is less than 70% it goes to wait diode
    return final_class



def capture_image():
    # Open the camera
    cap = cv2.VideoCapture(0)
    # Read a frame from the camera
    ret, frame = cap.read()
    # Release the camera
    cap.release()
    return frame

refPicture = capture_image()

def app(image_change_thres = 60):
    """
    For arduino use this function and add:

    import pyfirmata
    board = pyfirmata.Arduino('/dev/ttyACM0') # or other mount location
    board.digital[<pin id>].write(trash_class.value)
    """
    last_picture = refPicture
    while True:
        try:
            image = capture_image()
           # st.image(image)
            if last_picture is None or np.absolute(image.astype(int) - last_picture.astype(int)).max() > image_change_thres:
                if last_picture is not None and last_picture.any():
                    print(np.absolute(image.astype(int) - last_picture.astype(int)).max())
                    trash_class = process_image(image)
                    # board.digital[ < pin id >].write(trash_class.value) 
            last_picture = image
        except Exception as e:
            print(e)
        finally:
            time.sleep(2)
            
def diodes():
    return None

if __name__ == '__main__':
    app()
