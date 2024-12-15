import pyrealsense2 as rs
import numpy as np
import cv2
import os

def draw_axes(img, origin, imgpts):
    """
    Draw 3D coordinate axes on the image.

    :param img: The image to draw on.
    :param origin: The image coordinates of the origin (tuple).
    :param imgpts: The projected image coordinates of the axis points (numpy array, shape (3, 2)).
    :return: The image with the axes drawn.
    """
    imgpts = imgpts.astype(int)
    print(f"Origin (image coordinates): {origin}, type: {type(origin)}")
    for i, pt in enumerate(imgpts):
        pt_tuple = tuple(pt.ravel())
        print(f"Axis {i} point (image coordinates): {pt_tuple}, type: {type(pt_tuple)}")
        # Define axis colors: X - Red, Y - Green, Z - Blue
        color = [(0, 0, 255), (0, 255, 0), (255, 0, 0)][i]
        img = cv2.line(img, origin, pt_tuple, color, 3)
    return img

def load_vectors(rvec_path, tvec_path):
    """
    Load rotation vector and translation vector from .npy files.

    :param rvec_path: Path to the rotation vector file.
    :param tvec_path: Path to the translation vector file.
    :return: rvec and tvec as float32 numpy arrays, shape (3,1).
    """
    if not os.path.exists(rvec_path):
        raise FileNotFoundError(f"Rotation vector file not found: {rvec_path}")
    if not os.path.exists(tvec_path):
        raise FileNotFoundError(f"Translation vector file not found: {tvec_path}")

    rvec = np.load(rvec_path).astype(np.float32).squeeze()
    tvec = np.load(tvec_path).astype(np.float32).squeeze()

    print(f"Loaded rvec: {rvec}, shape: {rvec.shape}")
    print(f"Loaded tvec: {tvec}, shape: {tvec.shape}")

    # Reshape to (3,1) if needed
    rvec = rvec.reshape(3, 1) if rvec.shape == (3,) else rvec
    tvec = tvec.reshape(3, 1) if tvec.shape == (3,) else tvec

    print(f"Final rvec shape: {rvec.shape}")
    print(f"Final tvec shape: {tvec.shape}")
    print(f"Rotation vector (rvec):\n{rvec}")
    print(f"Translation vector (tvec):\n{tvec}")

    return rvec.astype(np.float32), tvec.astype(np.float32)

def load_distortion_coeffs(dist_coeffs_path):
    """
    Load distortion coefficients from a .npy file.
    If not found, assume no distortion.

    :param dist_coeffs_path: Path to the distortion coefficients file.
    :return: Distortion coefficients as a float32 numpy array.
    """
    if not os.path.exists(dist_coeffs_path):
        print(f"Distortion coefficients file not found: {dist_coeffs_path}. Assuming no distortion.")
        return np.zeros((5, 1), dtype=np.float32)  # Typically 5 coefficients
    dist_coeffs = np.load(dist_coeffs_path).astype(np.float32)
    print("Distortion coefficients loaded.")
    return dist_coeffs

def initialize_realsense():
    """
    Initialize RealSense pipeline, enable depth and color streams, and obtain camera intrinsics.

    :return: A tuple containing:
             - pipeline: RealSense pipeline object
             - depth_scale: Depth scale in meters/unit
             - intrinsics: RealSense camera intrinsics object
             - distortion_coeffs: Distortion coefficients (5x1)
    """
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable depth and color streams at 1280x720, 30 FPS
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Get depth scale
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # Get RGB camera intrinsics
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_stream.get_intrinsics()

    # Get distortion coefficients
    distortion_coeffs = np.array(intrinsics.coeffs, dtype=np.float32).reshape(5, 1)

    print(f"RealSense intrinsics:\n{intrinsics}")
    print(f"Distortion coefficients:\n{distortion_coeffs}")

    return pipeline, depth_scale, intrinsics, distortion_coeffs

def main():
    # Path to the folder containing calibration data
    image_folder = "./data"  # Replace with your data folder path
    rvec_path = os.path.join(image_folder, "rotation_vectors.npy")
    tvec_path = os.path.join(image_folder, "translation_vectors.npy")
    camera_matrix_path = os.path.join(image_folder, "rgb_intrinsic_matrix.npy")
    dist_coeffs_path = os.path.join(image_folder, "distortion_coeffs.npy")  # Distortion coefficients path

    # Load camera matrix
    if not os.path.exists(camera_matrix_path):
        raise FileNotFoundError(f"Camera matrix file not found: {camera_matrix_path}")
    camera_matrix = np.load(camera_matrix_path).astype(np.float32)
    print("Camera matrix (camera_matrix):\n", camera_matrix)

    # Load distortion coefficients
    dist_coeffs = load_distortion_coeffs(dist_coeffs_path)

    # Load rotation and translation vectors
    try:
        rvec, tvec = load_vectors(rvec_path, tvec_path)
        print("Rotation vector (rvec):\n", rvec)
        print("Translation vector (tvec):\n", tvec)
    except (FileNotFoundError, ValueError) as e:
        print(e)
        input("Press any key to exit...")
        return

    # Initialize RealSense pipeline
    pipeline, depth_scale, intrinsics_realsense, distortion_coeffs_realsense = initialize_realsense()

    # Create an alignment object to align depth frames to color frames
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Define axis length (in meters)
    axis_length = 0.2  # 0.2 meters

    # Define the 3D axes in world coordinates (X, Y, Z)
    axes_3d = np.float32([
        [axis_length, 0, 0],  # X-axis
        [0, axis_length, 0],  # Y-axis
        [0, 0, axis_length]   # Z-axis
    ])

    # Variables for sampling points when clicking on the image
    last_clicked_position = None
    last_clicked_world = None
    SAMPLE_COUNT = 10
    sampling = False
    sample_counter = 0
    sampled_points = []

    # Mouse callback function
    def mouse_callback(event, x, y, flags, param):
        nonlocal sampling, sample_counter, sampled_points, last_clicked_position, last_clicked_world
        if event == cv2.EVENT_LBUTTONDOWN:
            sampling = True
            sample_counter = 0
            sampled_points = []
            last_clicked_position = (x, y)
            last_clicked_world = None
            print(f"Mouse clicked at: ({x}, {y})")

    # Create a window and set mouse callback
    cv2.namedWindow('RealSense - World Axes')
    cv2.setMouseCallback('RealSense - World Axes', mouse_callback)

    try:
        while True:
            # Wait for frames
            frames = pipeline.wait_for_frames()

            # Align depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned depth and color frames
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                print("Invalid depth or color frame.")
                continue

            img_height, img_width = color_frame.height, color_frame.width

            # Convert color frame to numpy array
            color_image = np.asanyarray(color_frame.get_data())

            # Convert depth frame to numpy array (in meters)
            depth_image = np.asanyarray(depth_frame.get_data()).astype(float) * depth_scale

            # If in sampling mode, collect samples
            if sampling:
                if sample_counter < SAMPLE_COUNT:
                    x, y = last_clicked_position
                    print(f"Sampling {sample_counter + 1}/{SAMPLE_COUNT} point: (u={x}, v={y})")
                    if 0 <= x < img_width and 0 <= y < img_height:
                        z = depth_image[y, x]
                        if np.isfinite(z) and z > 0:
                            # Use RealSense deproject function
                            point_camera = rs.rs2_deproject_pixel_to_point(intrinsics_realsense, [x, y], z)
                            point_camera = np.array(point_camera, dtype=np.float32).reshape(3, 1)
                            sampled_points.append(point_camera)
                            print(f"Sampled camera coordinates: {point_camera.ravel()}")
                        else:
                            print("Invalid sampled depth value.")
                    else:
                        print("Sample point out of image range.")
                    sample_counter += 1
                else:
                    # Sampling done, compute average
                    if sampled_points:
                        avg_point_camera = np.mean(sampled_points, axis=0)  # shape (3,1)

                        # Convert rotation vector to rotation matrix
                        R, _ = cv2.Rodrigues(rvec)  # R transforms world to camera
                        R_inv = R.T  # R_inv transforms camera to world
                        t_wc = -R_inv @ tvec  # Translation from camera to world

                        # Transform camera coordinates to world coordinates
                        point_world = R_inv @ avg_point_camera + t_wc

                        print(f"Clicked pixel: (u={last_clicked_position[0]}, v={last_clicked_position[1]})")
                        print(
                            f"Camera coordinates (average): X={avg_point_camera[0][0]:.4f} m, "
                            f"Y={avg_point_camera[1][0]:.4f} m, Z={avg_point_camera[2][0]:.4f} m")
                        print(
                            f"World coordinates: X={point_world[0][0]:.4f} m, "
                            f"Y={point_world[1][0]:.4f} m, Z={point_world[2][0]:.4f} m")

                        # Store the last clicked world position
                        last_clicked_world = point_world.copy()
                    else:
                        print("No valid sampled points collected.")

                    # Reset sampling variables
                    sampling = False

            # Project 3D axes onto 2D image
            imgpts, _ = cv2.projectPoints(
                axes_3d,
                rvec,
                tvec,
                camera_matrix,
                dist_coeffs
            )
            imgpts = imgpts.astype(int).reshape(-1, 2)

            # Project the world origin (0,0,0)
            origin_world = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
            origin_img, _ = cv2.projectPoints(
                origin_world,
                rvec,
                tvec,
                camera_matrix,
                dist_coeffs
            )
            origin_img = tuple(origin_img.reshape(-1, 2).astype(int)[0])

            # Check if origin is within image bounds
            if not (0 <= origin_img[0] < img_width and 0 <= origin_img[1] < img_height):
                print("Origin is out of image range.")

            # Check if axis points are within image bounds
            for i, pt in enumerate(imgpts):
                if not (0 <= pt[0] < img_width and 0 <= pt[1] < img_height):
                    print(f"Axis point {i} {tuple(pt)} out of image range.")

            # Draw axes on the color image
            img_with_axes = color_image.copy()
            img_with_axes = draw_axes(img_with_axes, origin_img, imgpts)

            # Mark the origin
            cv2.circle(img_with_axes, origin_img, 5, (0, 0, 255), -1)  # Red dot

            # If we have a clicked point with world coordinates, redraw it
            if last_clicked_position and last_clicked_world is not None:
                x, y = last_clicked_position
                print(f"Redrawing the last clicked point: (u={x}, v={y})")
                cv2.circle(img_with_axes, (x, y), 5, (255, 255, 0), -1)  # Cyan dot
                cv2.putText(
                    img_with_axes,
                    f"X:{last_clicked_world[0][0]:.2f} Y:{last_clicked_world[1][0]:.2f} Z:{last_clicked_world[2][0]:.2f} m",
                    (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    2
                )

            # Process depth image for display (normalize to 0-255)
            depth_image_display = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_image_display = np.uint8(depth_image_display)
            depth_image_display = cv2.applyColorMap(depth_image_display, cv2.COLORMAP_JET)

            # Show images
            cv2.imshow('RealSense - World Axes', img_with_axes)
            cv2.imshow('Depth Image', depth_image_display)

            # Press 'q' or 'ESC' to quit
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                print("Press 'q' detected. Quitting the program.")
                break

    except KeyboardInterrupt:
        print("Program interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        input("Press any key to exit...")
    finally:
        # Stop the pipeline and close all windows
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
