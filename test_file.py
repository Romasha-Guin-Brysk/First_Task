import unittest
import cv2
from pathlib import Path
from First_Task import run_object_detection, det_model, det_ov_model

class TestObjectDetection(unittest.TestCase):
    
    def test_object_detection(self):
        print("Testing YOLO model with an image.")
        test_image_path = Path("First_Task/data/coco_bike.jpg")
        
        image = cv2.imread(str(test_image_path))
        self.assertIsNotNone(image, "Test image could not be loaded.")
        
        result = run_object_detection(str(test_image_path), det_model)
        
        self.assertIn("detections", result, "No detections found.")
        
        print("Test passed")
        
if __name__ == '__main__':
    unittest.main()
