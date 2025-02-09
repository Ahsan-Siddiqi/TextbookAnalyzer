import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

"""Captures images from camera using first image key s and second image key q."""
def capture_image():
    cap = cv2.VideoCapture(0)   # Change number to use different cameras

    if not cap.isOpened():
        print("Error: Could not open camera.")
    
    
    images = []
    for i in range(2):
        input(f"Press Enter to capture image {i + 1}...") 
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            images.append(None)
        else:
            images.append(frame)
    
    cap.release()
    return tuple(images)

"""SSIM comparison of two images."""
def compare(img1, img2):
    if img1 is not None and img2 is not None:
        # Code modified from https://stackoverflow.com/questions/56183201/detect-and-visualize-differences-between-two-images-with-opencv-python
        # Default channel grayscale, switched to multichannel for color comparison
        score = ssim(img1, img2, channel_axis=2)
        
        # Visualization of differences (not working atm - need to get second result from ssim)
        def compare_images(img1, img2):
            diff = (diff * 255).astype("uint8")
            diff_box = cv2.merge([diff, diff, diff])

            thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]

            mask = np.zeros(img1.shape, dtype='uint8')
            filled_after = img2.copy()

            for c in contours:
                area = cv2.contourArea(c)
                if area > 40:
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(img1, (x, y), (x + w, y + h), (36, 255, 12), 2)
                    cv2.rectangle(img2, (x, y), (x + w, y + h), (36, 255, 12), 2)
                    cv2.rectangle(diff_box, (x, y), (x + w, y + h), (36, 255, 12), 2)
                    cv2.drawContours(mask, [c], 0, (255, 255, 255), -1)
                    cv2.drawContours(filled_after, [c], 0, (0, 255, 0), -1)

            cv2.imshow('before', img1)
            cv2.imshow('after', img2)
            cv2.imshow('diff', diff)
            cv2.imshow('diff_box', diff_box)
            cv2.imshow('mask', mask)
            cv2.imshow('filled after', filled_after)
            cv2.waitKey()
        # compare_images(img1, img2)
    else:
        print("Error: Images were not captured.")
        score = None

    return score