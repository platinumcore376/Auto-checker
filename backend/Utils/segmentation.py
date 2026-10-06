import cv2
import numpy as np
import os
import shutil
import logging
from typing import Dict, List, Optional, Tuple
from config import (
    SEGMENTATION_MIN_HEIGHT,
    SEGMENTATION_PADDING,
    HISTOGRAM_LINE_THRESHOLD,
    DIAGRAM_MIN_WIDTH,
    DIAGRAM_MIN_HEIGHT,
    TEXT_CROP_PADDING,
)

logger = logging.getLogger("autochecker.segmentation")

class SegmentationError(Exception):
    """Base exception for line segmentation and diagram extraction errors."""
    pass

class NoTextDetectedError(SegmentationError):
    """Raised when no horizontal text lines can be identified in the image."""
    pass

def correct_tilt(image: np.ndarray) -> np.ndarray:
    """Corrects tilt in an image using Hough Line Transform."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

        if lines is not None:
            angles = []
            for line in lines:
                rho, theta = line[0]
                angle = theta * 180 / np.pi
                if 80 < angle < 100:
                    angles.append(90 - angle)

            if angles:
                avg_angle = float(np.mean(angles))
                h, w = image.shape[:2]
                center = (w // 2, h // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
                corrected_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
                return corrected_image
    except Exception as e:
        logger.warning(f"Tilt correction skipped due to warning: {str(e)}")
    return image

def extract_diagram(img_path: str, output_folder: str, sheet_idx: int = 0) -> Optional[str]:
    """
    Extracts diagram from negative text area image.
    Returns the file path of the saved diagram if found, or None if no diagram is present (BUG-05).
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"Cannot read image for diagram extraction: {img_path}")
            return None

        gray_no_text = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary_no_text = cv2.threshold(gray_no_text, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary_no_text, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        os.makedirs(output_folder, exist_ok=True)

        for idx, cnt in enumerate(contours[::-1]):
            x, y, w, h = cv2.boundingRect(cnt)
            if w > DIAGRAM_MIN_WIDTH and h > DIAGRAM_MIN_HEIGHT:
                diagram_img = img[y:y+h, x:x+w]
                diagram_filename = f"diagram_{sheet_idx}.png"
                diagram_path = os.path.join(output_folder, diagram_filename)
                cv2.imwrite(diagram_path, diagram_img)
                logger.info(f"Diagram detected and saved at {diagram_path}")
                return diagram_path

        logger.info(f"No valid diagram contours found in sheet {sheet_idx}")
        return None

    except Exception as e:
        logger.error(f"Diagram extraction failed on sheet {sheet_idx}: {str(e)}")
        return None

def visualize_text_region(
    img: np.ndarray,
    filtered_line_segments: List[Tuple[int, int]],
    output_folder: str,
    sheet_idx: int = 0
) -> Tuple[Optional[str], str]:
    """
    Crops the text region and produces a text-removed image for diagram extraction.
    Returns (text_crop_path, text_removed_image_path).
    """
    text_dir = os.path.join(output_folder, "texts")
    os.makedirs(text_dir, exist_ok=True)

    img_copy = img.copy()
    img_h, _ = img.shape[:2]

    text_removed_path = os.path.join(output_folder, "text_removed.png")
    text_crop_path = None

    if filtered_line_segments:
        min_y = min([seg[0] for seg in filtered_line_segments])
        max_y = max([seg[1] for seg in filtered_line_segments])

        start_y = max(min_y - TEXT_CROP_PADDING, 0)
        end_y = min(max_y + TEXT_CROP_PADDING, img_h)

        text_crop = img_copy[start_y:end_y, :]
        text_crop_path = os.path.join(text_dir, f"text_{sheet_idx}.png")
        cv2.imwrite(text_crop_path, text_crop)
        logger.info(f"Text crop saved to {text_crop_path}")

        # Mask text region with white background for diagram isolation
        img_copy[start_y:end_y, :] = (255, 255, 255)
        cv2.imwrite(text_removed_path, img_copy)
    else:
        logger.warning(f"No text line segments to visualize for sheet {sheet_idx}")
        cv2.imwrite(text_removed_path, img_copy)

    return text_crop_path, text_removed_path

def segment_lines_and_find_diagrams(
    img,
    output_folder: str,
    sheet_idx: int = 0,
    min_height_threshold: int = SEGMENTATION_MIN_HEIGHT,
    padding: int = SEGMENTATION_PADDING,
) -> Dict[str, Optional[str]]:
    """
    Performs tilt correction, horizontal line segmentation, text cropping, and diagram extraction.
    All outputs are saved strictly within output_folder (S1 request isolation).
    Returns a dictionary of result paths:
      {"segmented_folder": ..., "text_crop_path": ..., "diagram_path": ...}
    """
    try:
        # Support both NumPy array and PIL Image inputs
        if not isinstance(img, np.ndarray):
            img = np.array(img)

        # Convert to BGR for OpenCV processing
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        os.makedirs(output_folder, exist_ok=True)
        cv2.imwrite(os.path.join(output_folder, "original_image.png"), img)

        img = correct_tilt(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 10
        )

        hist = np.sum(binary, axis=1)
        threshold = HISTOGRAM_LINE_THRESHOLD
        lines = np.where(hist > threshold)[0]

        if len(lines) == 0:
            raise NoTextDetectedError(f"No horizontal text lines detected in sheet {sheet_idx}.")

        line_segments = []
        start = lines[0]
        for i in range(1, len(lines)):
            if lines[i] - lines[i - 1] > 10:
                line_height = lines[i - 1] - start
                if line_height >= min_height_threshold:
                    line_segments.append((start, lines[i - 1]))
                start = lines[i]

        if lines[-1] - start >= min_height_threshold:
            line_segments.append((start, lines[-1]))

        segmented_folder = os.path.join(output_folder, "segmented_lines")
        diagram_folder = os.path.join(output_folder, "diagrams")
        os.makedirs(segmented_folder, exist_ok=True)
        os.makedirs(diagram_folder, exist_ok=True)

        img_height = img.shape[0]
        filtered_line_segments = []

        for y_start, y_end in line_segments:
            line_segment = binary[y_start:y_end, :]
            if line_segment.shape[0] == 0:
                continue
            left_whitespace = np.sum(line_segment[:, :20] == 0) / (20 * line_segment.shape[0])
            right_whitespace = np.sum(line_segment[:, -20:] == 0) / (20 * line_segment.shape[0])

            if left_whitespace > 0.90 and right_whitespace < 0.96:
                filtered_line_segments.append((y_start, y_end))

        for idx, (y_start, y_end) in enumerate(filtered_line_segments):
            y_start_pad = max(0, y_start - TEXT_CROP_PADDING)
            y_end_pad = min(img_height, y_end + padding)
            line_img = img[y_start_pad:y_end_pad, :]
            cv2.imwrite(os.path.join(segmented_folder, f"line_{idx+1}.png"), line_img)

        logger.info(f"Segmented {len(filtered_line_segments)} text lines in sheet {sheet_idx}")

        text_crop_path, text_removed_path = visualize_text_region(
            img, filtered_line_segments, output_folder, sheet_idx=sheet_idx
        )

        diagram_path = extract_diagram(text_removed_path, diagram_folder, sheet_idx=sheet_idx)

        return {
            "segmented_folder": segmented_folder,
            "text_crop_path": text_crop_path,
            "diagram_path": diagram_path,
        }

    except NoTextDetectedError:
        raise
    except Exception as e:
        logger.error(f"Segmentation failed on sheet {sheet_idx}: {str(e)}")
        raise SegmentationError(f"Segmentation failed: {str(e)}") from e
