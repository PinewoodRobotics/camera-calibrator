import argparse
import json

import cv2 as cv
import platform
import numpy as np
import sys
from cscore import CameraServer


class OpenCvFrameReader:
    """Reads frames using OpenCV VideoCapture with configured resolution."""

    def __init__(self, port: int, resolution_width: int, resolution_height: int):
        self.port = port
        self.resolution_width = resolution_width
        self.resolution_height = resolution_height
        self.video_capture = None

    def open(self) -> None:
        self.video_capture = cv.VideoCapture(self.port)
        self.video_capture.set(cv.CAP_PROP_FRAME_WIDTH, self.resolution_width)
        self.video_capture.set(cv.CAP_PROP_FRAME_HEIGHT, self.resolution_height)

    def frames(self):
        if self.video_capture is None:
            self.open()
        while self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if not ret:
                break
            yield frame

    def close(self) -> None:
        if self.video_capture is not None:
            self.video_capture.release()


class WpilibFrameReader:
    """Reads frames using WPILib cscore CameraServer CvSink."""

    def __init__(
        self,
        port: int,
        resolution_width: int,
        resolution_height: int,
        set_fps: bool,
        requested_fps: int,
    ):
        self.port = port
        self.resolution_width = resolution_width
        self.resolution_height = resolution_height
        self.set_fps = set_fps
        self.requested_fps = requested_fps
        self.cv_sink = None
        self.frame_buffer = None

    def open(self) -> None:
        camera = CameraServer.startAutomaticCapture(dev=self.port)
        camera.setResolution(self.resolution_width, self.resolution_height)
        if self.set_fps:
            camera.setFPS(self.requested_fps)
        else:
            if platform.system() == "Darwin":
                try:
                    camera.setFPS(100)
                except Exception:
                    pass

        self.cv_sink = CameraServer.getVideo()
        self.frame_buffer = np.zeros(
            (self.resolution_height, self.resolution_width, 3), dtype=np.uint8
        )

    def frames(self):
        if self.cv_sink is None:
            self.open()
        while True:
            timestamp, frame = self.cv_sink.grabFrame(self.frame_buffer)
            if timestamp == 0:
                continue
            yield frame

    def close(self) -> None:
        # No explicit close needed for cscore objects in this context
        pass


class CharucoCapture:
    """Runs a capture-and-detect loop and accumulates detections for calibration."""

    def __init__(self, limit_fps: bool, detector: "cv.aruco.CharucoDetector"):
        self.limit_fps = limit_fps
        self.detector = detector

    def run(self, frame_source) -> tuple[list, list, list, tuple[int, int]]:
        frame_count = 0
        frame_shape = (0, 0)
        all_corners = []
        all_ids = []
        all_counter = []

        try:
            for frame in frame_source.frames():
                frame_count += 1
                if self.limit_fps and frame_count % 15 != 0:
                    continue

                frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                frame_shape = frame_gray.shape

                charuco_corners, charuco_ids, marker_corners, marker_ids = (
                    self.detector.detectBoard(frame_gray)
                )

                debug_image = frame

                if charuco_ids is not None and len(charuco_ids) >= 4:
                    all_corners.extend(charuco_corners)
                    all_ids.extend(charuco_ids)
                    all_counter.append(len(charuco_ids))

                    debug_image = cv.aruco.drawDetectedMarkers(
                        debug_image, marker_corners, marker_ids
                    )
                    debug_image = cv.aruco.drawDetectedCornersCharuco(
                        debug_image, charuco_corners, charuco_ids
                    )

                cv.putText(
                    debug_image,
                    f"Images captured: {len(all_counter)}",
                    (10, 30),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                cv.putText(
                    debug_image,
                    f"Resolution: {frame_shape[1]}x{frame_shape[0]}",
                    (10, 60),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                cv.imshow("Frame", debug_image)
                if cv.waitKey(1) == ord("q"):
                    break
        finally:
            frame_source.close()
            cv.destroyAllWindows()

        return all_corners, all_ids, all_counter, frame_shape


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
arg_parser.add_argument(
    "--use_wpilib", help="use WPILib cscore to open camera", action="store_true"
)
arg_parser.add_argument(
    "--set_fps",
    help="explicitly request device FPS (off by default; can crash on macOS)",
    action="store_true",
)

arg_parser.add_argument(
    "--resolution_width", help="resolution width", type=int, default=640
)
arg_parser.add_argument(
    "--resolution_height", help="resolution height", type=int, default=480
)

args = arg_parser.parse_args()

# Aruco Board
aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_1000)
charuco_board = cv.aruco.CharucoBoard(
    (args.width, args.height), args.square_size, args.marker_size, aruco_dict
)
charuco_detector = cv.aruco.CharucoDetector(charuco_board)

"""
Video capture and detection
"""

frame_reader = (
    WpilibFrameReader(
        port=args.port,
        resolution_width=args.resolution_width,
        resolution_height=args.resolution_height,
        set_fps=args.set_fps,
        requested_fps=args.fps,
    )
    if args.use_wpilib
    else OpenCvFrameReader(
        port=args.port,
        resolution_width=args.resolution_width,
        resolution_height=args.resolution_height,
    )
)

charuco_capture = CharucoCapture(limit_fps=args.limit_fps, detector=charuco_detector)
all_corners, all_ids, all_counter, frame_shape = charuco_capture.run(frame_reader)

# Calibrate
if (
    frame_shape == (0, 0)
    or len(all_counter) == 0
    or len(all_corners) == 0
    or len(all_ids) == 0
):
    print(
        "No valid detections captured. Ensure the Charuco board is visible and try again."
    )
    sys.exit(2)

if len(all_counter) < 3:
    print(
        f"Not enough images for calibration: got {len(all_counter)}, need at least 3."
    )
    sys.exit(2)

print("Calibrating camera...")
(
    _,
    camera_matrix,
    dist_coeffs,
    r_vecs,
    t_vecs,
    std_dev_intrinsics,
    std_dev_extrinsics,
    per_view_errors,
) = cv.aruco.calibrateCameraArucoExtended(
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
)

dist_coeffs = dist_coeffs.flatten()[:5].reshape(1, 5)

# Save calibration output
print("Saving calibration results...")
with open(args.output_file_path, "w") as f:
    camera_model = {
        "camera_matrix": camera_matrix.flatten().tolist(),
        "distortion_coefficients": dist_coeffs.flatten().tolist(),
        "avg_reprojection_error": np.average(np.array(per_view_errors)),
        "num_images": len(all_counter),
    }

    f.write(json.dumps(camera_model, indent=4))
print("Calibration complete!")
