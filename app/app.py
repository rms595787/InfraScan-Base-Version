from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import os

app = Flask(__name__)

# Serve the main HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Handle map click
@app.route('/map_click', methods=['POST'])
def map_click():
    lat = float(request.json.get('latitude'))
    lon = float(request.json.get('longitude'))

    # Define image paths (adjust these paths as needed)
    image_path1 = r"p1.png"
    image_path2 = r"p2.png"

    # Load images
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)

    # Proceed only if both images are loaded correctly
    if image1 is not None and image2 is not None:
        # Resize images to a common size
        resized_image1, resized_image2 = resize_images_to_common_size(image1, image2)

        # Compare images and get the difference
        diff_image_boxes, diff_image_highlighted, ssim_score = compare_images(resized_image1, resized_image2)

        # Save results
        result_dir = 'static/results'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        
        cv2.imwrite(os.path.join(result_dir, 'diff_image_boxes.png'), diff_image_boxes)
        cv2.imwrite(os.path.join(result_dir, 'diff_image_highlighted.png'), diff_image_highlighted)
        
        # Show images and differences
        show_images(resized_image1, resized_image2, diff_image_boxes, diff_image_highlighted, ssim_score)

        return jsonify({
            'status': 'success',
            'message': 'Image comparison completed.',
            'ssim_score': ssim_score,
            'result_path': result_dir
        })
    else:
        return jsonify({'status': 'error', 'message': 'Images could not be loaded.'})

def compare_images(image1, image2, min_area=1):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between the two images
    score, diff = ssim(gray1, gray2, full=True)
    diff = (diff * 255).astype("uint8")

    # Threshold the difference image
    _, thresh = cv2.threshold(diff, 200, 255, cv2.THRESH_BINARY_INV)

    # Find contours of the differences
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize images for displaying differences
    diff_image_boxes = image2.copy()
    diff_image_highlighted = np.zeros_like(image2)

    # Draw bounding boxes and highlight differences
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(diff_image_boxes, (x, y), (x + w, y + h), (0, 0, 255), 2)
            diff_image_highlighted[y:y+h, x:x+w] = image2[y:y+h, x:x+w]

    return diff_image_boxes, diff_image_highlighted, score

def show_images(image1, image2, diff_image_boxes, diff_image_highlighted, ssim_score):
    """
    Display the original images, difference images, and SSIM score.

    Args:
    image1 (ndarray): The first image.
    image2 (ndarray): The second image.
    diff_image_boxes (ndarray): Image with bounding boxes around differences.
    diff_image_highlighted (ndarray): Image with differences highlighted in white.
    ssim_score (float): SSIM score between the two images.
    """
    # Calculate similarity and difference percentages
    similarity_percentage = ssim_score * 100
    difference_percentage = 100 - similarity_percentage

    # Create a figure with a specified size
    plt.figure(figsize=(16, 12))

    # Display images in a 2x2 grid with padding
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    plt.title("Previous Image")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    plt.title("Current Image")
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(diff_image_boxes, cv2.COLOR_BGR2RGB))
    plt.title("Differences with Boxes")
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(diff_image_highlighted, cv2.COLOR_BGR2RGB))
    plt.title("Differences Highlighted With Black Color")
    plt.axis('off')

    # Adjust layout and padding
    plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05)
    plt.figtext(0.5, 0.01, f"SSIM Score: {ssim_score:.2f} | Similarity: {similarity_percentage:.2f}% | Difference: {difference_percentage:.2f}%", ha="center", fontsize=12)
    plt.show()

def resize_images_to_common_size(image1, image2):
    # Determine the minimum dimensions
    min_height = min(image1.shape[0], image2.shape[0])
    min_width = min(image1.shape[1], image2.shape[1])

    # Resize both images to the smallest dimensions
    resized_image1 = cv2.resize(image1, (min_width, min_height))
    resized_image2 = cv2.resize(image2, (min_width, min_height))

    return resized_image1, resized_image2

# Route to handle "Show Map" button click
@app.route('/show_map', methods=['POST'])
def show_map():
    # This will trigger the image comparison code
    return jsonify({'status': 'success', 'message': 'Image comparison triggered.'})

if __name__ == '__main__':
    app.run(debug=True)
