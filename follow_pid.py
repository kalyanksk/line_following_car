from GUI import GUI
from HAL import HAL
import cv2
import numpy as np 

kp = 0.005
Ki = 0.0001 
Kd = 0.0001  

prev_error = 0 
error_sum = 0  

# Create the initial mask
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])

while True:
    frame = HAL.getImage()
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
    # Find the contour of the line
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        line_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(line_contour)

        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # Get the center of the image
            image_center = frame.shape[1] // 2
            error = cX - image_center

            # Proportional control
            p_control = kp * error

            # Integral control
            error_sum += error
            i_control = Ki * error_sum

            # Derivative control
            d_control = Kd * (error - prev_error)
            prev_error = error

            # PID control action
            steering = p_control + i_control + d_control

            # Apply steering control
            HAL.setV(1)
            angular_velocity = -steering
            HAL.setW(angular_velocity)

        # Draw contour and centroid
        cv2.drawContours(frame, [line_contour], -1, (0, 255, 0), 2)
        cv2.circle(frame, (cX, cY), 5, (0, 255, 0), -1)

    # Display processed frame
    GUI.showImage(frame)

