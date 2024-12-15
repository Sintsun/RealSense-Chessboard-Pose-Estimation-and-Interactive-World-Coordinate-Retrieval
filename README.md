# RealSense Chessboard Pose Estimation and Interactive World Coordinate Retrieval

This repository contains a set of scripts that demonstrate how to:

- Use an Intel RealSense camera to capture synchronized color and depth images.
- Detect a chessboard in the scene and estimate the camera's pose (orientation and position) relative to the chessboard using `cv2.solvePnP`.
- Display and save the camera intrinsics, rotation, and translation vectors.
- Visualize reprojection error and draw coordinate axes in the scene.
- Interactively click on points in the image to retrieve their coordinates in both the camera and world coordinate systems, given a known camera pose.

## Features

### Chessboard Corner Detection and Pose Estimation
Using a known chessboard pattern, the script:
- Detects the 2D corners in the image.
- Uses the known 3D coordinates of these corners in the world frame.
- Applies `solvePnP` to compute the rotation and translation vectors (`rvec`, `tvec`) that describe the camera's pose relative to the chessboard.

### Camera Calibration Data Loading
Precomputed rotation vectors, translation vectors, camera intrinsics, and distortion coefficients are loaded from `.npy` files. This enables quick reuse of previously obtained calibration results.

### 3D Axes Visualization
Once the camera pose is known, the script projects a set of 3D axes (X, Y, Z) onto the 2D image, showing the camera's orientation and position relative to the chessboard.

### Depth-based 3D Point Retrieval
By clicking on a pixel in the image:
- The script samples depth data multiple times and averages it.
- Converts from 2D image coordinates + depth to 3D camera coordinates.
- Transforms these camera coordinates into world coordinates using the known camera pose.
- The resulting world coordinates are displayed on the image and printed to the console.

### Interactivity
- Press `s` in the initial calibration stage to save chessboard data.
- Press `q` in the main visualization stage to quit.
- Use the mouse to click on pixels in the image to retrieve their corresponding world coordinates.

## Requirements
- Python 3.x
- OpenCV (`cv2`)
- Intel RealSense SDK (`pyrealsense2`)
- A connected Intel RealSense camera

Make sure these packages are installed and that your Intel RealSense camera is properly connected.

## Usage

### Initial Chessboard Capture and Pose Estimation (If Needed)
1. Run the initial calibration script (e.g., `calibration.py`) to capture a chessboard image and compute the camera pose using `solvePnP`.
2. Once the corners are detected, press `s` to save the chessboard image, depth, and corners.
3. Saving this data also computes and stores `rvec`, `tvec`, and `rgb_intrinsic_matrix` in the `./data` folder.

### Pose Visualization and World Coordinate Retrieval
1. Ensure the calibration data (`rotation_vectors.npy`, `translation_vectors.npy`, `rgb_intrinsic_matrix.npy`, `distortion_coeffs.npy`) is available in the `./data` folder.
2. Run the main integrated script (e.g., `main.py`).
3. The script will:
   - Load the calibration and pose data.
   - Start the RealSense pipeline.
   - Continuously display:
     - A color image with 3D axes drawn at the world origin.
     - A depth image (colored for visualization).

### Clicking to Retrieve World Coordinates
- In the `RealSense - World Axes` window, click on a point.
- The script will sample the depth at that pixel multiple times and average it.
- It will then compute:
  - The 3D camera coordinate of the clicked point.
  - The corresponding 3D world coordinate using the known camera pose.
- The computed world coordinates (X, Y, Z) are displayed on the image and printed to the console.

### Exiting
- Press `q` or `ESC` in the main visualization window to quit the program.

## Acknowledgments
- Intel RealSense SDK
- OpenCV Library
