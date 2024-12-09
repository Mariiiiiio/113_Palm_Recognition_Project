import cv2
import mediapipe as mp
import os
from datetime import datetime
import keyboard
import time

class PalmImageCapture:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7
        )
        self.visible_camera = None
        self.nir_camera = None
        
    def initialize_cameras(self):
        # Assuming visible camera is index 0 and NIR camera is index 1
        self.visible_camera = cv2.VideoCapture(1)
        self.nir_camera = cv2.VideoCapture(0)
        
        if not self.visible_camera.isOpened() or not self.nir_camera.isOpened():
            raise Exception("Error: Cannot open one or both cameras")

    def detect_palm(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        return results.multi_hand_landmarks is not None

    def capture_images(self, folder_name, num_images=10):
        # Create folders for both modes
        visible_folder = os.path.join(folder_name, "visible")
        nir_folder = os.path.join(folder_name, "nir")
        os.makedirs(visible_folder, exist_ok=True)
        os.makedirs(nir_folder, exist_ok=True)

        images_captured = 0
        while images_captured < num_images:
            # Capture frames from both cameras
            ret_visible, frame_visible = self.visible_camera.read()
            ret_nir, frame_nir = self.nir_camera.read()

            if not ret_visible or not ret_nir:
                print("Error capturing frames")
                continue

            if self.detect_palm(frame_visible):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                
                # Save visible light image
                cv2.imwrite(os.path.join(visible_folder, f"visible_{timestamp}.jpg"), frame_visible)
                # Save NIR image
                cv2.imwrite(os.path.join(nir_folder, f"nir_{timestamp}.jpg"), frame_nir)
                
                images_captured += 1
                print(f"Captured image pair {images_captured}/{num_images}")
                time.sleep(0.5)  # Delay to avoid rapid capture

            # Display frames
            cv2.imshow("Visible Light", frame_visible)
            cv2.imshow("NIR", frame_nir)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def run(self):
        while True:
            name = input("Enter person's name (or 'quit' to exit): ")
            if name.lower() == 'quit':
                break

            folder_name = os.path.join("palm_images", name)
            os.makedirs(folder_name, exist_ok=True)

            while True:
                try:
                    self.initialize_cameras()
                    print("Place your palm in front of the camera...")
                    
                    # Wait for palm detection
                    while True:
                        ret, frame = self.visible_camera.read()
                        if ret and self.detect_palm(frame):
                            print("Palm detected! Press any key to start capturing...")
                            keyboard.read_event(suppress=True)
                            break
                        cv2.imshow("Waiting for palm", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    # Capture images
                    self.capture_images(folder_name)

                    # Ask for next action
                    print("\n1: Next Person")
                    print("2: Continue Capturing")
                    choice = input("Select an option: ")

                    if choice == "1":
                        break
                    elif choice != "2":
                        print("Invalid choice. Defaulting to next person.")
                        break

                finally:
                    # Clean up
                    if self.visible_camera:
                        self.visible_camera.release()
                    if self.nir_camera:
                        self.nir_camera.release()
                    cv2.destroyAllWindows()

        # Final cleanup
        self.hands.close()

if __name__ == "__main__":
    capture_system = PalmImageCapture()
    capture_system.run()
