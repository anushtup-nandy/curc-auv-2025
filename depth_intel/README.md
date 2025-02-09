
# **RealSense Object Detection with YOLO and 3D Positioning for CURC AUV - ROBOSUB2025**  

## **Overview**  
This project integrates **Intel RealSense** depth sensing with **YOLOv4** object detection to identify objects, determine their 3D coordinates, and assess proximity. Additionally, it extracts dominant colors and estimates camera orientation based on detected objects.

## **Features**  
- **Object Detection**: Utilizes OpenCV's DNN module with YOLOv4 for real-time object detection.  
- **Depth Estimation**: Extracts 3D coordinates using **Intel RealSense SDK**.  
- **Color Recognition**: Determines the dominant color of detected objects.  
- **Proximity Warning**: Highlights objects within a **1-meter range**.  
- **Camera Orientation Estimation**: Computes **roll, pitch, and yaw** based on object positions.

---

## **Installation**  

### **Prerequisites**  
Ensure you have the following dependencies installed:  
- **Python 3.7+**  
- **OpenCV** (`cv2`)  
- **Intel RealSense SDK (`pyrealsense2`)**  
- **NumPy**  
- **YOLOv4 model files** (`yolov4.cfg`, `yolov4.weights`, `coco.names`)  

### **Setup**  

1. **Clone the Repository**  
   ```sh
   git clone https://github.com/your-repo/realsense-yolo.git
   cd realsense-yolo
   ```

2. **Install Dependencies**  
   ```sh
   pip install opencv-python numpy pyrealsense2
   ```

3. **Download YOLOv4 Weights & Config Files**  
   Download `yolov4.weights`, `yolov4.cfg`, and `coco.names` and place them in the project directory.

4. **Run the Script**  
   ```sh
   python object_detection.py
   ```

---

## **How It Works**  

### **1. RealSense Camera Initialization**  
- Streams both **color (RGB)** and **depth** frames.  
- Retrieves the **depth scale** for accurate 3D position estimation.

### **2. Object Detection using YOLOv4**  
- The script loads YOLOv4’s model and **detects objects** in the color image.  
- It applies **non-maximum suppression (NMS)** to filter overlapping detections.  
- Detected objects are assigned **bounding boxes** with labels.

### **3. 3D Position Estimation**  
- The depth frame is used to **convert 2D pixel coordinates** into **3D world coordinates**.  
- The script retrieves the object's **X, Y, Z coordinates**.

### **4. Dominant Color Extraction**  
- Uses **K-Means clustering** to determine the most dominant color.  
- Converts the **BGR** values into **HSV** to classify colors.

### **5. Proximity Warning System**  
- Objects within **1 meter** are marked in **red**, while others remain **green**.

### **6. Camera Orientation Estimation**  
- Computes **roll, pitch, and yaw** based on detected objects.

---

## **Example Output**  
When an object is detected, the output includes:  
- Bounding box with **label and confidence**.  
- **3D coordinates (X, Y, Z) in meters**.  
- **Color classification** (e.g., Red, Blue, Green).  
- **Proximity warning** (if within 1m).  
- **Camera angles (Roll, Pitch, Yaw)**.

---

## **Controls**  
- **Press 'q'** to exit the program.

---

## **Future Enhancements**  
- Add **multi-camera support** for better depth accuracy.  
- Integrate **ROS** for robotic applications.  
- Optimize **real-time performance** using GPU acceleration.

---

## **Acknowledgments**  
- **Intel RealSense SDK** for depth sensing.  
- **YOLOv4** for object detection.  
- **OpenCV** for image processing.  

---

## **License**  
..