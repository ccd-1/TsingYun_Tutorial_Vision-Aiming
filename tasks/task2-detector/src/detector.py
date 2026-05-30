"""Task 2 MNIST-board detector helpers with student TODO extension points.

This file belongs to Task 2. The simulator runner imports it so that a Task 2
implementation can be tested both offline and inside the Unity simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from simulator_client.protocol import Matrix3x3
from model import classify_mnist_digit

Point2D = tuple[float, float]
CornerSet = tuple[Point2D, Point2D, Point2D, Point2D]
RgbPixel = tuple[int, int, int]
ImageLike = np.ndarray
WARP_OUTPUT_SIZE = 128
MNIST_INNER_RATIO = 0.69


@dataclass(frozen=True)
class BoundingBox:
    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> Point2D:
        return (self.x + self.width * 0.5, self.y + self.height * 0.5)


@dataclass
class Detection:
    class_id: int
    confidence: float
    bbox: BoundingBox
    corners: CornerSet
    rvec: object | None = None
    tvec: object | None = None


def _bbox_from_corners(corners: Sequence[Point2D]) -> BoundingBox:
    if len(corners) != 4:
        raise ValueError(f"Expected 4 corners, got {len(corners)}")

    xs = [float(point[0]) for point in corners]
    ys = [float(point[1]) for point in corners]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    return BoundingBox(
        x=min_x,
        y=min_y,
        width=max_x - min_x + 1.0,
        height=max_y - min_y + 1.0,
    )


def _crop_bounds(corners: Sequence[Point2D], image_width: int, image_height: int) -> tuple[int, int, int, int]:
    bbox = _bbox_from_corners(corners)
    x0 = max(0, min(image_width, int(np.floor(bbox.x))))
    y0 = max(0, min(image_height, int(np.floor(bbox.y))))
    x1 = max(0, min(image_width, int(np.ceil(bbox.x + bbox.width))))
    y1 = max(0, min(image_height, int(np.ceil(bbox.y + bbox.height))))
    return x0, y0, x1, y1


def crop_bbox(image: np.ndarray, corner_candidates: Sequence[Sequence[Point2D]]) -> list[np.ndarray]:
    crops: list[np.ndarray] = []
    for corners in corner_candidates:
        if len(corners) != 4:
            continue

        # `corners` are expected in LU, RU, RD, LD order.
        src = np.array(corners, dtype=np.float32)

        # Shrink the source quad toward its center so the warp removes the outer red border.
        center = np.mean(src, axis=0)
        src = center + (src - center) * MNIST_INNER_RATIO

        dst = np.array(
            [
                [0, 0],
                [WARP_OUTPUT_SIZE - 1, 0],
                [WARP_OUTPUT_SIZE - 1, WARP_OUTPUT_SIZE - 1],
                [0, WARP_OUTPUT_SIZE - 1],
            ],
            dtype=np.float32,
        )

        perspective = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(
            image,
            perspective,
            (WARP_OUTPUT_SIZE, WARP_OUTPUT_SIZE),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        crops.append(warped)
    return crops


def order_corners(corners: Sequence[Point2D]) -> CornerSet:
    if len(corners) != 4:
        raise ValueError(f"Expected 4 corners, got {len(corners)}")

    pts = np.asarray(corners, dtype=np.float32)
    sums = pts[:, 0] + pts[:, 1]
    diffs = pts[:, 0] - pts[:, 1]

    top_left = tuple(pts[np.argmin(sums)].tolist())
    bottom_right = tuple(pts[np.argmax(sums)].tolist())
    top_right = tuple(pts[np.argmin(diffs)].tolist())
    bottom_left = tuple(pts[np.argmax(diffs)].tolist())

    return (top_left, top_right, bottom_right, bottom_left)


def detect_bbox(image: ImageLike, threshold: int = 200) -> list[CornerSet]:
    if image is None:
        return []

    if image.ndim == 3 and image.shape[2] >= 3:
        channels = image[..., :3].astype(np.int16)
        c0, c1, c2 = channels[..., 0], channels[..., 1], channels[..., 2]
        red_mask = (
            ((c0 > c1 + 40) & (c0 > c2 + 40) & (c0 >= threshold))
            | ((c2 > c1 + 40) & (c2 > c0 + 40) & (c2 >= threshold))
        )
    else:
        red_mask = image.astype(np.int16) >= threshold

    red_mask = red_mask.astype(np.uint8) * 255
    kernel = np.ones((5, 5), dtype=np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    corner_candidates: list[CornerSet] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon=0.02 * perimeter, closed=True)
        if approx.shape[0] != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue

        corners = [tuple(pt[0].astype(float).tolist()) for pt in approx]
        ordered = order_corners(corners)

        xs = [p[0] for p in ordered]
        ys = [p[1] for p in ordered]
        width = float(max(xs) - min(xs))
        height = float(max(ys) - min(ys))
        if width <= 0 or height <= 0:
            continue

        aspect_ratio = width / height if height > 0 else 0.0
        if not 0.3 <= aspect_ratio <= 3.0:
            continue

        corner_candidates.append(ordered)

    return corner_candidates


def detect_mnist_board(image: ImageLike, threshold: int = 200) -> list[Detection]:
    corner_candidates = detect_bbox(image, threshold=threshold)
    if not corner_candidates:
        return []

    crops = crop_bbox(np.asarray(image, dtype=np.uint8), corner_candidates)
    detections: list[Detection] = []

    for corners, crop in zip(corner_candidates, crops):
        class_id, confidence = classify_mnist_digit(crop)
        if confidence < 0.0 or class_id < 0:
            continue

        bbox = _bbox_from_corners(corners)
        detections.append(
            Detection(
                class_id=class_id,
                confidence=float(confidence),
                bbox=bbox,
                corners=corners,
            )
        )

    return detections


def solve_pnp(
    detections: Sequence[Detection],
    camera_matrix: Matrix3x3,
    board_width_meters: float,
    board_height_meters: float,
    dist_coeffs: Sequence[float] | None = None,
) -> list[Detection]:
    if camera_matrix is None:
        raise ValueError("camera_matrix is required for solve_pnp")

    half_width = board_width_meters / 2.0
    half_height = board_height_meters / 2.0
    object_points = np.array(
        [
            [-half_width, -half_height, 0.0],
            [half_width, -half_height, 0.0],
            [half_width, half_height, 0.0],
            [-half_width, half_height, 0.0],
        ],
        dtype=np.float32,
    )

    camera_array = np.asarray(camera_matrix, dtype=np.float64)
    if dist_coeffs is None:
        dist_array = np.zeros((4, 1), dtype=np.float64)
    else:
        dist_array = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)

    result: list[Detection] = []
    for detection in detections:
        image_points = np.array(detection.corners, dtype=np.float32).reshape((4, 2))
        success, rvec, tvec = cv2.solvePnP(
    object_points, image_points, camera_array, dist_array, flags=cv2.SOLVEPNP_ITERATIVE
)
        if not success:
            continue

        detection.rvec = rvec
        detection.tvec = tvec
        result.append(detection)

    return result
