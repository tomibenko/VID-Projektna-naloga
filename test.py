import unittest
from flask import Flask, json
from unittest.mock import patch
import numpy as np
import cv2

# Assuming your Flask app and functions are in a file named `app.py`
from data_managing import *

class FlaskTestCase(unittest.TestCase):

    def setUp(self):
        # Creates a test client
        self.app = app.test_client()
        # Propagate the exceptions to the test client
        self.app.testing = True
    
    def test_random_augment_image(self):
        # Create a dummy image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Call the function under test
        augmented_img = random_augment_image(img)
        # Check if the output is still an image (basic check)
        self.assertIsInstance(augmented_img, np.ndarray)


if __name__ == '__main__':
    unittest.main()