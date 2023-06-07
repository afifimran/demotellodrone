import cv2
import numpy as np
import time
from djitellopy import Tello
# b54b
#set points (center of the frame coordinates in pixels)
rifX = 960/2
rifY = 720/2

#PI constant
Kp_X = 0.1
Ki_X = 0.0
Kp_Y = 0.2
Ki_Y = 0.0

#Loop time
Tc = 0.05

#PI terms initialized
integral_X = 0
error_X = 0
previous_error_X = 0
integral_Y = 0
error_Y = 0
previous_error_Y = 0

centroX_pre = rifX
centroY_pre = rifY

net = cv2.dnn.readNetFromCaffe("C:\\Users\\DroneTello\\PycharmProjects\\tello-gesture-control\\Human-Detection-and-Tracking-through-the-DJI-Tello-minidrone\\MobileNetSSD_deploy.prototxt.txt", "C:\\Users\\DroneTello\\PycharmProjects\\tello-gesture-control\\Human-Detection-and-Tracking-through-the-DJI-Tello-minidrone\\MobileNetSSD_deploy.caffemodel") #modify with the NN path

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]


colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))


drone = Tello()  # declaring drone object
time.sleep(2.0) #waiting 2 seconds
print("Connecting...")
drone.connect()
print("BATTERY: ")
print(drone.get_battery())
time.sleep(1.0)
print("Loading...")
drone.streamon()  # start camera streaming
print("Takeoff...")
drone.takeoff()

count = 0
count2 = 0
while True:
	start = time.time()
	frame = drone.get_frame_read().frame


	cv2.circle(frame, (int(rifX), int(rifY)), 1, (0,0,255), 10)
	h,w,channels = frame.shape

	blob = cv2.dnn.blobFromImage(frame,
		0.007843, (180, 180), (0,0,0),True, crop=False)

	net.setInput(blob)
	detections = net.forward()


	for i in np.arange(0, detections.shape[2]):

		idx = int(detections[0, 0, i, 1])
		confidence = detections[0, 0, i, 2]

		if CLASSES[idx] == "chair" and confidence > 0.5:

			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				colors[idx], 2)
			#draw the center of the person detected
			centroX = (startX + endX)/2
			centroY = (2*startY + endY)/3

			# print(centroX)
			# print(centroY)

			centroX_pre = centroX
			centroY_pre = centroY

			# print(centroX_pre)
			# print(centroX_pre)


			cv2.circle(frame, (int(centroX), int(centroY)), 1, (0,0,255), 10)

			error_X = -(rifX - centroX)
			error_Y = rifY - centroY

			# print("errorx1")
			# print(error_X)
			# print("errory1")
			# print(error_Y)

			# cv2.line(frame, (int(rifX),int(rifY)), (int(centroX),int(centroY)), (0,255,255),5 )


			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)

			# print(y)
			
			#PI controller
			integral_X = integral_X + error_X*Tc # updating integral PID term
			uX = Kp_X*error_X + Ki_X*integral_X # updating control variable uX
			previous_error_X = error_X # update previous error variable

			# print(integral_X)
			# print(uX)
			# print(previous_error_X)
			
			integral_Y = integral_Y + error_Y*Tc # updating integral PID term
			uY = Kp_Y*error_Y + Ki_Y*integral_Y
			previous_error_Y = error_Y

			# print(integral_Y)
			# print(uY)
			# print(previous_error_Y)
			if count==2:
				drone.send_rc_control(0,0,round(uY),round(uX))
				if uX<=5 and uX>=-5 or uY<=5 and uY>=-5 :
					drone.send_rc_control(0, 5, round(uY), round(uX))

				count=0
			#drone.send_rc_control(0, 0, round(uY), round(uX))
			#time.sleep(1.0)
			#break when a person is recognized
			count=count+1
			# print("count")
			# print(count)
			break	


		else: #if nobody is recognized take as reference centerX and centerY of the previous frame
			centroX = centroX_pre
			centroY = centroY_pre
			cv2.circle(frame, (int(centroX), int(centroY)), 1, (0,0,255), 10)

			error_X = -(rifX - centroX)
			error_Y = rifY - centroY

			# print("errorx2")
			# print(error_X)
			# print("errory3")
			# print(error_Y)

			# cv2.line(frame, (int(rifX),int(rifY)), (int(centroX),int(centroY)), (0,255,255),5 )

			integral_X = integral_X + error_X*Tc # updating integral PID term
			uX = Kp_X*error_X + Ki_X*integral_X # updating control variable uX
			previous_error_X = error_X # update previous error variable
			
			integral_Y = integral_Y + error_Y*Tc # updating integral PID term
			uY = Kp_Y*error_Y + Ki_Y*integral_Y
			previous_error_Y = error_Y

			if count2==2:
				# print("rc2")
				drone.send_rc_control(0,0,round(uY),round(uX))
				if uX<=5 and uX>=-5 or uY<=5 and uY>=-5 :
					# print("HERE")
					drone.send_rc_control(0, -10, round(uY), round(uX))

				count2=0

			count2 = count2 + 1
			# print("count2")
			# print(count2)
			# time.sleep(1.0)

			# print(round(uY))
			# print(round(uX))

			continue

	
	cv2.imshow("Frame", frame)

	end = time.time()
	elapsed= end-start
	if Tc - elapsed > 0:
		time.sleep(Tc - elapsed)
	end_ = time.time()
	elapsed_ = end_ - start
	fps = 1/elapsed_
	print("FPS: ",fps)


	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

drone.streamoff()
cv2.destroyAllWindows()
drone.land()
print("Landing...")
print("BATTERY: ")
print(drone.get_battery())
drone.end()
