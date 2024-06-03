import unittest
import os
import projekt_1 as projekt
import tempfile

class TestProjekt(unittest.TestCase):

    def test_capture_images(self):
        with tempfile.TemporaryDirectory() as tempdir:
            projekt.capture_images(output_dir=tempdir, num_images=5)
            self.assertEqual(len(os.listdir(tempdir)), 5)

    def test_create_model(self):
        model = projekt.create_model()
        self.assertEqual(model.count_params(), 417002)  # Expected number of parameters

    def test_memory_usage(self):
        usage_before = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
        projekt.print_memory_usage()
        usage_after = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
        self.assertGreater(usage_after, usage_before)

if __name__ == '__main__':
    unittest.main()
