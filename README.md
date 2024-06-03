# Water-Level
This is a repo on a water level monitoring system that is done just by running an algorithm and is connected to an external camera. There is also a Human Machine Interface Included.

This algorithm uses the edge detection method to calculate the water level that is in a specific water tank. How it works is by changing the picture into grayscale then find the largest contour from the grayscale image.

How to use: 
install OpenCV to use the code
install uvicorn, websockets and FastAPI to use the Human Machine Interface
