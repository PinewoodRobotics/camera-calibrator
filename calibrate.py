import argparse
import json

import cv2 as cv
import numpy as np
from tqdm import tqdm

# Parse CLI arguments
arg_parser = argparse.ArgumentParser(prog="calibrate")
arg_parser.add_argument("--port", help="port of input camera", type=int)
arg_parser.add_argument("--output_file_path", help="path to output file", type=str)
arg_parser.add_argument("--square_size", help="width of square in meters", type=float)
arg_parser.add_argument("--marker_size", help="width of marker in meters", type=float)
arg_parser.add_argument("--width", help="width of board", type=int)
arg_parser.add_argument("--height", help="height of board", type=int)
arg_parser.add_argument("--limit_fps", help="limit fps", type=bool, default=False)
arg_parser.add_argument("--fps", help="fps", type=int, default=50)

arg_parser.add_argument("--resolution_width", help="resolution width", type=int, default=640)
arg_parser.add_argument("--resolution_height", help="resolution height", type=int, default=480)

args = arg_parser.parse_args()

# Aruco Board
aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_1000)
charuco_board = cv.aruco.CharucoBoard((args.width, args.height), args.square_size, args.marker_size, aruco_dict)
charuco_detector = cv.aruco.CharucoDetector(charuco_board)

# Video capture
video_capture = cv.VideoCapture(args.port)
video_capture.set(cv.CAP_PROP_FRAME_WIDTH, args.resolution_width)
video_capture.set(cv.CAP_PROP_FRAME_HEIGHT, args.resolution_height)
video_capture.set(cv.CAP_PROP_FPS, args.fps)

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

    if args.limit_fps and frame_count % 15 != 0:
        continue

    # Detect
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    frame_shape = frame_gray.shape

    charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(frame_gray)

    debug_image = frame

    if charuco_ids is not None and len(charuco_ids) >= 4:
        all_corners.extend(charuco_corners)
        all_ids.extend(charuco_ids)
        all_counter.append(len(charuco_ids))

        debug_image = cv.aruco.drawDetectedMarkers(debug_image, marker_corners, marker_ids)
        debug_image = cv.aruco.drawDetectedCornersCharuco(debug_image, charuco_corners, charuco_ids)

    cv.putText(debug_image, f"Images captured: {len(all_counter)}", (10, 30), 
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv.imshow("Frame", debug_image)
    if cv.waitKey(1) == ord("q"):
        break

video_capture.release()
cv.destroyAllWindows()

# Calibrate
print("Calibrating camera...")
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
print("Saving calibration results...")
with open(args.output_file_path, "w") as f:
    camera_model = {
        "camera_matrix": camera_matrix.flatten().tolist(),
        "distortion_coefficients": dist_coeffs.flatten().tolist(),
        "avg_reprojection_error": np.average(np.array(per_view_errors)),
        "num_images": len(all_counter)
    }

    f.write(json.dumps(camera_model, indent=4))
print("Calibration complete!")
