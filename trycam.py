import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recog = cv2.face.LBPHFaceRecognizer_create()
recog.read('trainer.yml')

lbls = {}
with open('labels.pickle', 'rb') as f:
	tlbls = pickle.load(f)
	lbls = {v:k for k,v in tlbls.items()}
	
cap = cv2.VideoCapture(0)

while(True):
	# start camera
	ret, frame = cap.read()
	
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #gray filter
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5) #detect face
	
	for (x,y,w,h) in faces:
		#print(x,y,w,h) #show face location
		
		
		#specify image save
		face_gray = gray[y:y+h, x:x+w]
		face_color = frame[y:y+h, x:x+w]
		
		id_, acc = recog.predict(face_gray)
		print(acc)
		if acc>=50:
			print(lbls[id_])
		if acc>=80:
			font = cv2.FONT_HERSHEY_SIMPLEX
			name = lbls[id_]
			color = (0,0,0)
			stroke = 3
			cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
		
		img_item = "pic.png"
		cv2.imwrite(img_item, face_gray)
			
		#mark face
		color = (0,0,255)
		stroke = 2
		xend = x+w
		yend = y+h
		cv2.rectangle(frame, (x,y), (xend, yend), color, stroke)
	
	# Display frame
	cv2.imshow('frame',frame)
	k = cv2.waitKey(20)
    
	#esc button to exit
	if k==27:
		break

# release the capture when finished
cap.release()
cv2.destroyAllWindows()