import cv2 as cv
from cv2 import aruco
import numpy as np
import os

DIR = os.path.dirname(__file__).replace("\\", "/")

# load in the calibration data
calib_data_path = f"{DIR}/camera_calibration_realsensed435/camera_mat_distort.npz"

calib_data = np.load(calib_data_path)
print(calib_data.files)

camera_mat = calib_data["camera_mat"]
camera_distort = calib_data["camera_distort"]
camera_rvecs = calib_data["camera_rvecs"]
camera_tvecs = calib_data["camera_tvecs"]

MARKER_SIZE = 10.6  # centimeters (measure your printed marker size)

marker_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)

param_markers = aruco.DetectorParameters()

cap = cv.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    marker_corners, marker_IDs, reject = aruco.detectMarkers(
        gray_frame, marker_dict, parameters=param_markers
    )
    if marker_corners:
        aruco_rvecs, aruco_tvecs, _ = aruco.estimatePoseSingleMarkers(
            marker_corners, MARKER_SIZE, camera_mat, camera_distort
        )
        total_markers = range(0, marker_IDs.size)
        for ids, corners, i in zip(marker_IDs, marker_corners, total_markers):
            
            # draw closed lines of the marker
            cv.polylines( # see details in cv2.polylines(): https://shengyu7697.github.io/python-opencv-polylines/
                frame, # img
                [corners.astype(np.int32)], # points
                True, # closed or not
                (0, 255, 255), # color
                3, # thickness
                cv.LINE_AA, # linetype
            )

            corners = corners.reshape(4, 2)
            corners = corners.astype(int)
            top_right = corners[0].ravel()
            top_left = corners[1].ravel()
            bottom_right = corners[2].ravel()
            bottom_left = corners[3].ravel()

            # draw corners of the marker
            # ul corner
            cv.circle(
                frame, # img
                corners[0], # center of the circle
                5, # radius of the circle
                (0, 0, 255), # color
                -1 # negative means filled circle
                )
            
            # ur corner
            cv.circle(
                frame, # img
                corners[1], # center of the circle
                5, # radius of the circle
                (0, 0, 255), # color
                -1 # negative means filled circle
                )
            
            # br corner
            cv.circle(
                frame, # img
                corners[2], # center of the circle
                5, # radius of the circle
                (0, 0, 255), # color
                -1 # negative means filled circle
                )
            
            # br corner
            cv.circle(
                frame, # img
                corners[3], # center of the circle
                5, # radius of the circle
                (0, 0, 255), # color
                -1 # negative means filled circle
                )


            # Calculating the distance
            distance = np.sqrt(
                aruco_tvecs[i][0][0] ** 2 + aruco_tvecs[i][0][1] ** 2 + aruco_tvecs[i][0][2] ** 2
            )

            # Draw the pose of the marker
            point = cv.drawFrameAxes(frame, camera_mat, camera_distort, aruco_rvecs[i], aruco_tvecs[i], 4, 4)
            
            # Marker ID
            cv.putText( # see details in cv2.putText(): https://shengyu7697.github.io/python-opencv-puttext/
                frame, # img
                f"ID: {ids[0]}", # text
                top_right + (0, -20), # org, also try (320,240) [pix], etc.
                cv.FONT_HERSHEY_COMPLEX_SMALL , # fontFace, also try .FONT_HERSHEY_SIMPLEX
                1, # fontScale
                (0, 0, 255), # color
                1, # thickness
                cv.LINE_AA, # lineType
            )

            # Marker position
            cv.putText(
                frame, 
                f"position (x-y-z)= ({round(aruco_tvecs[i][0][0],2)}, {round(aruco_tvecs[i][0][1],2)}, {round(aruco_tvecs[i][0][2],2)})",
                bottom_left + (0, 30),
                cv.FONT_HERSHEY_COMPLEX_SMALL,
                0.7,
                (0, 0, 255),
                1,
                cv.LINE_AA,
            )

            # Marker position
            cv.putText(
                frame, 
                f"orientation (3D-vector)= ({round(aruco_rvecs[i][0][0],2)}, {round(aruco_rvecs[i][0][1],2)}, {round(aruco_rvecs[i][0][2],2)})",
                bottom_left + (0, 50),
                cv.FONT_HERSHEY_COMPLEX_SMALL,
                0.7,
                (0, 0, 255),
                1,
                cv.LINE_AA,
            )

            # print(ids, "  ", corners)

    cv.imshow("frame", frame)
    key = cv.waitKey(1)
    if key == ord("q"):
        break
cap.release()
cv.destroyAllWindows()
