import json
from pathlib import Path

import cv2
import numpy as np

TASK_ROOT = Path(__file__).resolve().parents[1]

# TODO(student): fill in your own calibration image folder.
# choose the folder that contains your chessboard calibration photos
# keep this as a pathlib Path so list_calibration_images can search it
CALIBRATION_IMAGES_DIR = TASK_ROOT / "data" / "calibration"

# TODO(student): change this if your image extension is different.
# examples: "*.jpg", "*.png", or "*.jpeg"
# use one glob pattern at a time so the input order stays easy to inspect
CALIBRATION_IMAGE_GLOB = "*.jpg"

# TODO(student): fill in your own calibration target information.
# set PATTERN_SIZE to the number of inner chessboard corners, not square count
# measure one square side length in meters and store it in SQUARE_SIZE_METERS
CALIBRATION_TARGET_TYPE = "chessboard"
PATTERN_SIZE = (-1, -1)
SQUARE_SIZE_METERS = -1

CAMERA_PARAMS_PATH = TASK_ROOT / "output" / "camera_params.json"


def list_calibration_images():
    return sorted(Path(CALIBRATION_IMAGES_DIR).glob(CALIBRATION_IMAGE_GLOB))


def create_board_points(pattern_size, square_size_meters):
    cols, rows = pattern_size
    points = np.zeros((rows * cols, 3), dtype=np.float32)
    for row in range(rows):
        for col in range(cols):
            index = row * cols + col
            x = col * square_size_meters
            y = row * square_size_meters
            z = 0 
            points[index] = (x, y, z)
    return points 


def detect_calibration_points(gray_image, pattern_size):
    if gray_image is None:
        return False, np.empty((0, 2), dtype=np.float32)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray_image, pattern_size, flags=flags)
    if not found:
        return False, np.empty((0, 2), dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    window_size = (11, 11)  
    dead_zone = (-1, -1)
    refined = cv2.cornerSubPix(gray_image, corners, window_size, dead_zone, criteria)
    return True, refined


def _is_valid_calibration_result(result):
    if result is None:
        return False
    if not isinstance(result, tuple) or len(result) != 2:
        return False

    camera_matrix, dist_coeffs = result
    camera_matrix = np.asarray(camera_matrix)
    dist_coeffs = np.asarray(dist_coeffs)
    return (
        camera_matrix.shape == (3, 3)
        and dist_coeffs.size >= 4
        and np.all(np.isfinite(camera_matrix))
        and np.all(np.isfinite(dist_coeffs))
        and abs(float(camera_matrix[2, 2])) > 1e-12
    )


def calibrate_camera(object_points, image_points, image_size):
    # TODO(student): Run camera calibration from all matched 3D / 2D points.
    # call cv2.calibrateCamera with all object/image point pairs
    # read rms reprojection error, camera_matrix, and dist_coeffs
    # if OpenCV fails or returns non-finite values:
    #     raise a clear error
    # return camera_matrix and dist_coeffs
    if not object_points or not image_points:
        raise ValueError("No object_points or image_points provided for calibration")

    obj_pts = [np.asarray(op, dtype=np.float32) for op in object_points]
    img_pts = [np.asarray(ip, dtype=np.float32) for ip in image_points]

    if image_size is None:
        raise ValueError("image_size must be provided")

    flags = 0
    try:
        rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, image_size, None, None, flags=flags)
    except cv2.error as e:
        raise RuntimeError(f"OpenCV calibrateCamera failed: {e}")

    if not _is_valid_calibration_result((camera_matrix, dist_coeffs)):
        raise RuntimeError("Calibration produced invalid camera matrix or distortion coefficients")

    return camera_matrix, dist_coeffs


def save_camera_params(camera_matrix, dist_coeffs, image_size, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.reshape(-1).tolist(),
    }
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main():
    image_paths = list_calibration_images()
    if not image_paths:
        raise SystemExit(f"No calibration images found in {CALIBRATION_IMAGES_DIR} matching {CALIBRATION_IMAGE_GLOB}")

    board_points = create_board_points(PATTERN_SIZE, SQUARE_SIZE_METERS)

    object_points = []
    image_points = []
    image_size = None

    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_size = (gray.shape[1], gray.shape[0])

        found, points = detect_calibration_points(gray, PATTERN_SIZE)
        if not found:
            print(f"Skip image without valid points: {image_path.name}")
            continue

        object_points.append(board_points.copy())
        image_points.append(points)
        print(f"Use image: {image_path.name}")

    if not object_points:
        raise SystemExit("No valid calibration images were collected.")

    camera_matrix, dist_coeffs = calibrate_camera(object_points, image_points, image_size)
    save_camera_params(camera_matrix, dist_coeffs, image_size, CAMERA_PARAMS_PATH)

    print(f"Saved camera parameters to: {CAMERA_PARAMS_PATH}")


if __name__ == "__main__":
    main()
