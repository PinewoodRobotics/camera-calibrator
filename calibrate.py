import sys
import json

import cv2 as cv
import numpy as np

GRID_WIDTH = 11
GRID_HEIGHT = 8

SQUARE_WIDTH_M = 0.018
MARKER_WIDTH_M = 0.014

# Aruco Board
aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_1000)
charuco_board = cv.aruco.CharucoBoard((GRID_WIDTH, GRID_HEIGHT), SQUARE_WIDTH_M, MARKER_WIDTH_M, aruco_dict)
charuco_detector = cv.aruco.CharucoDetector(charuco_board)

# Video capture
video_capture = cv.VideoCapture(sys.argv[1])
frame_count = 0
frame_shape = (0, 0)

# Detection output
all_corners = []
all_ids = []
all_counter = []

while video_capture.isOpened():
    ret, frame = video_capture.read()

    if not ret:
        break

    # Limit FPS
    frame_count += 1

    if frame_count % 15 != 0:
        continue

    # Detect
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    frame_shape = frame_gray.shape

    charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(frame_gray)

    if len(charuco_ids) >= 4:
        all_corners.extend(charuco_corners)
        all_ids.extend(charuco_ids)
        all_counter.append(len(charuco_ids))

    debug_image = cv.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)
    debug_image = cv.aruco.drawDetectedCornersCharuco(debug_image, charuco_corners, charuco_ids)

    cv.imshow("Frame", debug_image)
    if cv.waitKey(1) == ord('q'):
        break

video_capture.release()
cv.destroyAllWindows()

# Calibrate
_, camera_matrix, dist_coeffs, r_vecs, t_vecs, std_dev_intrinsics, std_dev_extrinsics, per_view_errors = cv.aruco.calibrateCameraArucoExtended(
    np.array(all_corners),
    np.array(all_ids),
    np.array(all_counter),
    charuco_board,
    frame_shape,
    np.array([]),
    np.array([]),
    np.array([]),
    np.array([]),
    np.array([]),
    np.array([]),
    np.array([]),
    cv.CALIB_RATIONAL_MODEL)

# Save calibration output
with open(sys.argv[2], "w") as f:
    camera_model = {
        "camera_matrix": camera_matrix.flatten().tolist(),
        "distortion_coefficients": dist_coeffs.flatten().tolist(),
        "avg_reprojection_error": np.average(np.array(per_view_errors)),
        "num_images": len(all_counter)
    }

    f.write(json.dumps(camera_model, indent=4))
