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
        self.roi_size = (640, 1024)  # Region of Interest size
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
        
        # Make the frame square
        palm_img = frame[y_min:y_max, x_min:x_max]
        
        # Resize to standard size if needed
        if palm_img.size > 0:
            palm_img = cv2.resize(palm_img, self.roi_size)
            
        # Draw guide rectangle on display frame
        display_frame = frame.copy()
        cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        return palm_img, display_frame, True
    
    def capture_images(self, folder_path):
        # Create a subfolder for 640x640 images
        folder_path_640 = os.path.join(folder_path, "640imgs")
        os.makedirs(folder_path_640, exist_ok=True)
        
        images_captured = 0
        
        while images_captured < 10:
            # Capture from both cameras
            ret_visible, visible_frame = self.visible_cam.read()
            ret_nir, nir_frame = self.nir_cam.read()
            
            if not ret_visible or not ret_nir:
                continue
            
            # Process both frames
            cropped_visible, display_frame_visible, palm_visible = self.detect_and_crop_palm(visible_frame)
            cropped_nir, display_frame_nir, palm_nir = self.detect_and_crop_palm(nir_frame)
            
            # Combine displays side by side
            combined_display = np.hstack((display_frame_visible, display_frame_nir))
            
            # Display guide and status
            text = f"Remaining: {10-images_captured}"
            cv2.putText(combined_display, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined_display, "VISIBLE", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined_display, "NIR", (display_frame_visible.shape[1] + 10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow("Camera Feed", combined_display)
            
            key = cv2.waitKey(1)
            if (key == ord(' ') and palm_visible and palm_nir and 
                cropped_visible is not None and cropped_nir is not None):
                # Save both cropped palm images
                timestamp = datetime.now().strftime("%H%M%S_%f")
                
                # Save original size images
                vis_filename = f"{folder_path}/visible_{timestamp}.jpg"
                nir_filename = f"{folder_path}/nir_{timestamp}.jpg"
                cv2.imwrite(vis_filename, cropped_visible)
                cv2.imwrite(nir_filename, cropped_nir)
                self.visible_images.append(vis_filename)
                self.nir_images.append(nir_filename)
                
                # Save 640x640 resized images
                resized_visible = cv2.resize(cropped_visible, (640, 640))
                resized_nir = cv2.resize(cropped_nir, (640, 640))
                
                vis_filename_640 = f"{folder_path_640}/visible_{timestamp}.jpg"
                nir_filename_640 = f"{folder_path_640}/nir_{timestamp}.jpg"
                cv2.imwrite(vis_filename_640, resized_visible)
                cv2.imwrite(nir_filename_640, resized_nir)
                
                images_captured += 1
            
            if key == 27:  # ESC to exit
                break
            
        cv2.destroyAllWindows()
        return images_captured == 10
    
    def create_blended_images(self, folder_path):
        """Create blended images from visible and NIR captures in matching order"""
        if len(self.visible_images) != 10 or len(self.nir_images) != 10:
            print("Error: Missing images for blending")
            return
        
        print("\nCreating blended images...")
        
        # Ensure 640imgs folder exists
        folder_path_640 = os.path.join(folder_path, "640imgs")
        os.makedirs(folder_path_640, exist_ok=True)
        
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
            
            # Create blended image (original size)
            blended = cv2.addWeighted(nir_img, 0.6, visible_img, 0.4, -25)
            
            # Create 640x640 version
            blended_640 = cv2.resize(blended, (640, 640))
            
            # Extract timestamp from visible image path for consistent naming
            timestamp = vis_path.split('_')[-1]  # Get timestamp part
            
            # Save both versions
            cv2.imwrite(f"{folder_path}/blended_{timestamp}", blended)
            cv2.imwrite(f"{folder_path_640}/blended_{timestamp}", blended_640)
        
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
                print("\nCapturing visible light/NIR images...")
                visible_success = self.capture_images(folder_path)
                
                # print("\nCapturing NIR images...")
                # nir_success = self.capture_images(folder_path)
                
                if visible_success:
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
