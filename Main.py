import cv2
import numpy as np

# Initialize video capture
# If using external camera then change the value to cv2.VideoCapture(1)
cap = cv2.VideoCapture(0)

# Define initial bounding box values (adjust as needed)
x1, y1, x2, y2 = 100, 100, 300, 400

# Define a flag to indicate if the bounding box is being adjusted
adjusting_bbox = False

# Define the kernel size for adjusting the accuracy
kernel_size = 1

# Define the height of the bounding box in meters (adjust as needed)
bbox_height_meters = 2.0

# Define the low and high level thresholds (adjust as needed)
low_level_threshold = 0.2  # meters
high_level_threshold = 1.6  # meters

while True:
    # Capture the current frame
    ret, frame = cap.read()

    # Draw the bounding box on the frame
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Create a mask where the bounding box is white and the rest is black
    mask = np.zeros_like(frame)
    cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)

    # Apply the mask to the frame
    cropped_frame = cv2.bitwise_and(frame, mask)

    # Convert the cropped frame to grayscale
    gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

    # Blur the grayscale image for better edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection to the blurred image to detect the edges and can adjust the threshold
    edges = cv2.Canny(image=blurred, threshold1=55, threshold2=70)

    # Define the kernel for morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Open the edges to remove small objects
    opened = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

    # Close the edges to fill in small gaps
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    # Find the contours in the edge detected image
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Sort the contours by area and grab the largest one
        largest_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

        # Calculate the bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.putText(frame, f"x: {x}, y: {y}, w: {w}, h: {h}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Get the coordinates of the contour points within the bounding box
        contour_points_in_bbox = []
        for point in largest_contour:
            point_x, point_y = point[0]
            if x1 <= point_x < x2 and y1 <= point_y < y2:
                contour_points_in_bbox.append((point_x - x1, point_y - y1))

        # Find the y-coordinate of the lowest and highest points in the contour
        lowest_y = min(point[1] for point in contour_points_in_bbox)
        highest_y = max(point[1] for point in contour_points_in_bbox)

        # Calculate the water level as a percentage of the bounding box height
        water_level_percentage = ((highest_y + 1) / (y2 - y1)) * 100

        # Calculate the water level in meters
        water_level_meters = (bbox_height_meters * (100 - water_level_percentage)) / 100

        # Display the water level in meters on top left of window
        cv2.putText(frame, f"Water Level: {water_level_meters:.2f}m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Print low or high level based on the thresholds
        if water_level_meters < low_level_threshold:
            print("Low Level")
        elif water_level_meters > high_level_threshold:
            print("High Level")
    else:
        print("No contours found") 

    # Display the original frame, the cropped frame, and the edge detected frame
    cv2.imshow('Original', frame)
    cv2.imshow('Cropped', cropped_frame)
    cv2.imshow('Edges', edges)

    # Handle user input for adjusting the bounding box
    c = cv2.waitKey(1)
    if c == ord('a'):  # Adjust the bounding box
        adjusting_bbox = True
        print("Use the mouse to adjust the bounding box (click and drag).")
    elif c == ord('r'):  # Reset the bounding box
        x1, y1, x2, y2 = 100, 100, 300, 400
    elif c == 27:  # Escape key to exit
        break

    # Adjust the bounding box based on mouse events
    if adjusting_bbox:
        def on_mouse(event, x, y, flags, param):
            global x1, y1, x2, y2
            if event == cv2.EVENT_LBUTTONDOWN:
                x1, y1 = x, y
            elif event == cv2.EVENT_LBUTTONUP:
                x2, y2 = x, y
                adjusting_bbox = False
                print("Bounding box adjusted.")

        cv2.setMouseCallback('Original', on_mouse)

# Release the video capture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
