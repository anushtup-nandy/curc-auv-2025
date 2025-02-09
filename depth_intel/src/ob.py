import cv2
import pyrealsense2 as rs
import numpy as np
import math

# Initialize RealSense Pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
profile = pipeline.start(config)

# Get depth scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# Alternative object detection using OpenCV DNN
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

def get_3d_coordinates(depth_frame, pixel_x, pixel_y):
    """Convert pixel coordinates to 3D coordinates using depth information"""
    depth = depth_frame.get_distance(int(pixel_x), int(pixel_y))
    depth_intrin = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    point_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, [pixel_x, pixel_y], depth)
    return point_3d

def get_dominant_color(image, x, y, w, h):
    """Extract the dominant color from the detected object region with improved ranges"""
    # Ensure coordinates are within image boundaries
    height, width = image.shape[:2]
    x = max(0, min(x, width-1))
    y = max(0, min(y, height-1))
    w = min(w, width - x)
    h = min(h, height - y)
    
    if w <= 0 or h <= 0:
        return (0, 0, 0), 'unknown'
    
    # Extract ROI and apply preprocessing
    roi = image[y:y+h, x:x+w]
    if roi.size == 0:
        return (0, 0, 0), 'unknown'
    
    # Apply bilateral filter to reduce noise while preserving edges
    roi = cv2.bilateralFilter(roi, 9, 75, 75)
    
    # Convert to different color spaces for better analysis
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lab_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    
    # Reshape and combine color spaces
    pixels = np.float32(roi.reshape(-1, 3))
    hsv_pixels = np.float32(hsv_roi.reshape(-1, 3))
    
    # Use k-means with multiple clusters
    n_colors = 3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    
    # Get the dominant color (most frequent cluster)
    _, counts = np.unique(labels, return_counts=True)
    dominant_idx = np.argmax(counts)
    dominant_color = centers[dominant_idx].astype(np.uint8)
    
    # Enhanced color ranges with more variations
    color_ranges = {
        'red': [
            ([0, 100, 100], [10, 255, 255]),   # Red range 1
            ([160, 100, 100], [180, 255, 255]) # Red range 2
        ],
        'blue': [
            ([100, 50, 50], [130, 255, 255])   # Blue range
        ],
        'green': [
            ([40, 50, 50], [80, 255, 255])     # Green range
        ],
        'yellow': [
            ([20, 100, 100], [35, 255, 255])   # Yellow range
        ],
        'orange': [
            ([10, 100, 100], [20, 255, 255])   # Orange range
        ],
        'purple': [
            ([130, 50, 50], [160, 255, 255])   # Purple range
        ]
    }
    
    # Convert dominant color to HSV for range checking
    dominant_hsv = cv2.cvtColor(np.uint8([[dominant_color]]), cv2.COLOR_BGR2HSV)[0][0]
    
    # Check color ranges
    color_name = 'unknown'
    max_confidence = 0
    
    for name, ranges in color_ranges.items():
        for (lower, upper) in ranges:
            lower = np.array(lower)
            upper = np.array(upper)
            
            # Calculate confidence based on how well the color fits in the range
            confidence = np.mean(np.logical_and(
                dominant_hsv >= lower,
                dominant_hsv <= upper
            ).astype(float))
            
            if confidence > max_confidence and confidence > 0.5:
                max_confidence = confidence
                color_name = name
    
    return dominant_color, color_name

try:
    while True:
        # Wait for frames
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Object detection using OpenCV DNN
        height, width, _ = color_image.shape
        blob = cv2.dnn.blobFromImage(color_image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Information to display
        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-maximum suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                
                # Get 3D coordinates
                center_x = x + w//2
                center_y = y + h//2
                point_3d = get_3d_coordinates(depth_frame, center_x, center_y)
                distance = np.sqrt(sum([x*x for x in point_3d]))
                
                # Get color information
                dominant_color, color_name = get_dominant_color(color_image, x, y, w, h)

                # Display with confidence
                cv2.putText(color_image,
                            f"{label} ({color_name})",
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)
                
                # Draw bounding box
                cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Display object information including color
                cv2.putText(color_image, 
                           f"{label} ({color_name}): {distance:.2f}m",
                           (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (0, 255, 0), 2)
                
                # Display 3D coordinates
                cv2.putText(color_image,
                           f"X:{point_3d[0]:.2f} Y:{point_3d[1]:.2f} Z:{point_3d[2]:.2f}",
                           (x, y + h + 20),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (0, 255, 0), 2)
                
                # Display color swatch
                cv2.rectangle(color_image, 
                            (x + w + 10, y), 
                            (x + w + 30, y + 20), 
                            dominant_color.tolist(), 
                            -1)

        # Display the image
        cv2.imshow('Object Detection', color_image)
        
        # Break loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()