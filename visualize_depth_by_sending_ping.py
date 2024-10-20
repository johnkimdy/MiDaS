import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time

# Continuous sending of colored line
def send_colored_line(input_img):
    # Load the depth map image in grayscale
    depth_map = cv2.imread(input_img, cv2.IMREAD_GRAYSCALE)

    # Apply a binary threshold to segment the image
    _, binary_image = cv2.threshold(depth_map, 50, 255, cv2.THRESH_BINARY_INV)

    # Use Canny edge detection to find edges
    edges = cv2.Canny(binary_image, 100, 200)

    # Find contours from the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))

    # Filter contours based on area
    contour_depths = []
    for contour in contours:
        if cv2.contourArea(contour) < 10:  # Adjust threshold as needed
            continue
        
        # Compute the average depth of the contour
        avg_depth = np.mean(depth_map[contour[:, 0, 1], contour[:, 0, 0]])
        contour_depths.append((contour, avg_depth))

    # Sort contours by average depth (shallow to deep)
    contour_depths.sort(key=lambda x: x[1])

    # Create a color version of the grayscale depth map
    color_depth_map = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2RGB)

    while True:
        for contour, avg_depth in contour_depths:
            # Draw the current contour in red
            cv2.drawContours(color_depth_map, [contour], -1, (255, 0, 0), 2)
            
            # Show the depth map with the current contour
            plt.imshow(color_depth_map)
            plt.axis('off')  # Hide axes
            plt.pause(0.01)  # Brief pause to update the plot
            plt.clf()  # Clear the figure for the next plot

            time.sleep(1)  # Wait for 1 second before showing the next contour

            # Optional: Reset the color depth map for the next iteration
            color_depth_map = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2RGB)

        # Optionally break after one complete iteration through the contours
        # break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_img',
                        default=None,
                        help='Path to the input grayscale depth map image'
                        )
    
    args = parser.parse_args()

    # Start sending the colored line
    send_colored_line(args.input_img)