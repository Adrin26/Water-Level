import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture device")
    exit()

# Define initial bounding box values (adjust as needed)
x1, y1, x2, y2 = 100, 100, 300, 400

# Define a flag to indicate if the bounding box is being adjusted
adjusting_bbox = False

# Define the kernel size for morphological operations
kernel_size = 1

# Define the height of the bounding box in meters (adjust as needed)
bbox_height_meters = 2.0

# WebSocket connection manager
connections = []

# Serve the HTML file
@app.get("/", response_class=HTMLResponse)
def get_html():
    try:
        with open("client2.html", "r") as f:
            html = f.read()
        return HTMLResponse(content=html, status_code=200)
    except Exception as e:
        print(f"Error reading client2.html: {e}")
        return HTMLResponse(content="Error reading client.html", status_code=500)

@app.get("/read_image", response_class=Response)
async def read_image():
    return FileResponse("static/utm.png")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connections.append(websocket)
    print(f"New WebSocket connection accepted: {websocket}")

    try:
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

            # Apply Canny edge detection to the blurred image
            edges = cv2.Canny(image=blurred, threshold1=180, threshold2=250)

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
                cv2.putText(frame, f"x: {x}, y: {y}, w: {w}, h: {h}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

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

                # Check if the water level is below 0.4 or above 1.6 meters
                if water_level_meters < 0.4:
                    alert_message = f"Warning: Water level is low ({water_level_meters:.2f}m)"
                elif water_level_meters > 1.6:
                    alert_message = f"Warning: Water level is high ({water_level_meters:.2f}m)"
                else:
                    alert_message = None

                # Display the water level and alert message (if any)
                cv2.putText(frame, f"Water Level: {water_level_meters:.2f}m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # Uncomment below to display the error message in the display frame
   #             if alert_message:
   #                cv2.putText(frame, alert_message, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            else:
                print("No contours found")
                alert_message = None

            # Encode the alert message as bytes
            alert_message_bytes = alert_message.encode() if alert_message else b''

            # Encode the frame as JPEG
            _, encoded_frame = cv2.imencode('.jpg', frame)

            # Concatenate the alert message and frame data, separated by a null character
            data_to_send = alert_message_bytes + b'\0' + encoded_frame.tobytes()

            # Send the concatenated data to all connected WebSocket clients
            for connection in connections:
                try:
                    await connection.send_bytes(data_to_send)
                except Exception as e:
                    print(f"Error sending data: {e}")
                    connections.remove(connection)

    except WebSocketDisconnect:
        connections.remove(websocket)
        print(f"WebSocket connection closed: {websocket}")

# Run the FastAPI application with Uvicorn
if __name__ == "__main__":
    print("Starting server...")
    try:
        uvicorn.run(app, host="127.0.0.1", port=8000)
    except Exception as e:
        print(f"Error running server: {e}")