import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime

class PalmCaptureSystem:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7
        )
        
        # Initialize cameras
        self.visible_cam = cv2.VideoCapture(0)
        self.nir_cam = cv2.VideoCapture(1)
        
        # Set camera properties for consistent capture
        for cam in [self.visible_cam, self.nir_cam]:
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
        # Frame settings
        self.roi_size = (640, 640)  # Region of Interest size
        self.margin = 50  # Extra margin for palm cropping
        
        # Add path attributes for storing image paths
        self.visible_images = []
        self.nir_images = []
    
    def create_folder(self, name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_path = f"palm_images/{name}_{timestamp}"
        os.makedirs(folder_path, exist_ok=True)
        return folder_path
    
    def detect_and_crop_palm(self, frame):
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if not results.multi_hand_landmarks:
            return None, frame, False
        
        # Get palm landmarks
        landmarks = results.multi_hand_landmarks[0]
        h, w = frame.shape[:2]
        
        # Calculate palm bounding box
        x_coords = [lm.x * w for lm in landmarks.landmark]
        y_coords = [lm.y * h for lm in landmarks.landmark]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # Add margin
        x_min = max(0, x_min - self.margin)
        y_min = max(0, y_min - self.margin)
        x_max = min(w, x_max + self.margin)
        y_max = min(h, y_max + self.margin)
        
        # Crop palm region
        palm_img = frame[y_min:y_max, x_min:x_max]
        
        # Resize to standard size if needed
        if palm_img.size > 0:
            palm_img = cv2.resize(palm_img, self.roi_size)
            
        # Draw guide rectangle on display frame
        display_frame = frame.copy()
        cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        return palm_img, display_frame, True
    
    def capture_images(self, folder_path, mode="visible"):
        camera = self.visible_cam if mode == "visible" else self.nir_cam
        images_captured = 0
        image_paths = []  # Store paths of captured images
        
        while images_captured < 10:
            ret, frame = camera.read()
            if not ret:
                continue
            
            cropped_palm, display_frame, palm_detected = self.detect_and_crop_palm(frame)
            
            # Display guide and status
            text = f"{mode.upper()}: {10-images_captured} remaining"
            cv2.putText(display_frame, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow("Camera Feed", display_frame)
            
            key = cv2.waitKey(1)
            if key == ord(' ') and palm_detected and cropped_palm is not None:
                # Save cropped palm image
                timestamp = datetime.now().strftime("%H%M%S_%f")
                filename = f"{folder_path}/{mode}_{timestamp}.jpg"
                cv2.imwrite(filename, cropped_palm)
                image_paths.append(filename)  # Store the path
                images_captured += 1
            
            if key == 27:  # ESC to exit
                break
        
        # Store paths based on mode
        if mode == "visible":
            self.visible_images = image_paths
        else:
            self.nir_images = image_paths
            
        cv2.destroyAllWindows()
        return images_captured == 10
    
    def create_blended_images(self, folder_path):
        """Create blended images from visible and NIR captures in matching order"""
        if len(self.visible_images) != 10 or len(self.nir_images) != 10:
            print("Error: Missing images for blending")
            return
        
        print("\nCreating blended images...")
        
        # Sort images by timestamp in filename
        self.visible_images.sort()
        self.nir_images.sort()
        
        for vis_path, nir_path in zip(self.visible_images, self.nir_images):
            # Read both images
            visible_img = cv2.imread(vis_path)
            nir_img = cv2.imread(nir_path)
            
            # Ensure both images exist and have the same size
            if visible_img is None or nir_img is None:
                continue
                
            if visible_img.shape != nir_img.shape:
                nir_img = cv2.resize(nir_img, (visible_img.shape[1], visible_img.shape[0]))
            
            # Create blended image (50% visible, 50% NIR)
            blended = cv2.addWeighted(nir_img, 0.7, visible_img, 0.3, 25)
            
            # Extract timestamp from visible image path for consistent naming
            timestamp = vis_path.split('_')[-1]  # Get timestamp part
            filename = f"{folder_path}/blended_{timestamp}"
            cv2.imwrite(filename, blended)
        
        # Clear the stored paths
        self.visible_images = []
        self.nir_images = []
        
        print("Blended images created successfully!")
    
    def run(self):
        while True:
            name = input("Enter person's name (or 'quit' to exit): ")
            if name.lower() == 'quit':
                break
            
            folder_path = self.create_folder(name)
            
            while True:
                print("\nCapturing visible light images...")
                visible_success = self.capture_images(folder_path, "visible")
                
                print("\nCapturing NIR images...")
                nir_success = self.capture_images(folder_path, "nir")
                
                if visible_success and nir_success:
                    print("\nCapture complete!")
                    self.create_blended_images(folder_path)
                
                choice = input("\n1: Next Person\n2: Continue Capturing\nChoice: ")
                if choice == '1':
                    break
                elif choice == '2':
                    continue

        
        self.visible_cam.release()
        self.nir_cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_system = PalmCaptureSystem()
    capture_system.run()
