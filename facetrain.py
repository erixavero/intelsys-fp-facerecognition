import os
from PIL import Image
import numpy as np
import cv2
import pickle

#scan folder for images
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
imgdir = os.path.join(BASE_DIR,'images')

#for training faces
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
ylabels = [] #collect names
xtrain =[] #collect face data

#face recognizer
recog = cv2.face.LBPHFaceRecognizer_create()

#make face id
curID = 0
labelID = {}

print('reading pics...')
for root, dirs, files in os.walk(imgdir):
	for file in files:
		if file.endswith("png") or file.endswith("jpg"):
			path = os.path.join(root, file)
			label = os.path.basename(os.path.dirname(path))
			#print(path, label)
			if not label in labelID:
				labelID[label] = curID
				curID = curID+1
			idtolabel = labelID[label]
			
			#turn images to numbers
			pilimg = Image.open(path).convert('L') #turn img grayscale
			size = (500,500) #resize image
			fimg = pilimg.resize(size, Image.ANTIALIAS)
			
			img_arr = np.array(fimg, 'uint8')
			#print(img_arr)
			
			faces = face_cascade.detectMultiScale(img_arr)
			for (x,y,w,h) in faces:
				tgt = img_arr[y:y+h,x:x+w]
				xtrain.append(tgt)
				ylabels.append(idtolabel)
				
print('xtrain ', xtrain)
print('ylabel ',ylabels)
print('labelID',labelID)

print('training...')
with open('labels.pickle', 'wb') as f:
	pickle.dump(labelID,f)
	
recog.train(xtrain, np.array(ylabels))
recog.save('trainer.yml')
print('done')
