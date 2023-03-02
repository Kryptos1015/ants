import cv2
import imutils

# grayscale, blur
ants_img = cv2.imread("ants3.jpg")
gray = cv2.cvtColor(ants_img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# contours
c = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
c = imutils.grab_contours(c)

# points and unique id
for i, c in enumerate(c):
    
    M = cv2.moments(c)
    if M["m00"] != 0:
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        cv2.circle(ants_img, (cx, cy), 2, (0, 0, 255), -1)
        cv2.putText(ants_img, f"Ant {i+1}", (cx - 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

cv2.imshow("Ants Detected", ants_img)
cv2.waitKey(0)
