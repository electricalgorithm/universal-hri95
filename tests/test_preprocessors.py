"""
The unit-test for the preprocessor functions.
"""
from unittest.mock import patch
import numpy
import pytest

from harmonicsradius.preprocessors import (
    shrink_to,
    linear_upscale,
    bicubic_upscale,
    nearest_upscale
)


class TestPreprocessors:
    """The unit-test for the preprocessor functions."""

    def test_shrink_to(self):
        """Test the shrink_to function."""
        image = numpy.zeros((100, 100, 3), numpy.uint8)
        new_image = shrink_to(image, 50, 50)
        assert new_image.shape == (50, 50, 3)
        # Test if cv2.INTER_LINEAR is called.
        with patch("cv2.resize") as mock_resize:
            shrink_to(image, 50, 50)
            mock_resize.assert_called_with(image, (50, 50),
                                           interpolation=3)

    def test_linear_upscale(self):
        """Test the linear_upscale function."""
        image = numpy.zeros((100, 100, 3), numpy.uint8)
        new_image = linear_upscale(image, 2)
        assert new_image.shape == (200, 200, 3)
        #  Test if cv2.INTER_LINEAR is called.
        with patch("cv2.resize") as mock_resize:
            linear_upscale(image, 2)
            mock_resize.assert_called_with(image, (0, 0), fx=2, fy=2,
                                           interpolation=1)

    def test_bicubic_upscale(self):
        """Test the bicubic_upscale function."""
        image = numpy.zeros((100, 100, 3), numpy.uint8)
        new_image = bicubic_upscale(image, 2)
        assert new_image.shape == (200, 200, 3)
        # Test if cv2.INTER_CUBIC is called.
        with patch("cv2.resize") as mock_resize:
            bicubic_upscale(image, 2)
            mock_resize.assert_called_with(image, (0, 0), fx=2, fy=2,
                                           interpolation=2)

    @pytest.mark.parametrize("factor", [2, 3, 4])
    def test_nearest_upscale(self, factor):
        """Test the nearest_upscale function."""
        image = numpy.zeros((100, 100, 3), numpy.uint8)
        new_image = nearest_upscale(image, factor)
        assert new_image.shape == (factor*100, factor*100, 3)
        # Test if cv2.INTER_NEAREST is called.
        with patch("cv2.resize") as mock_resize:
            nearest_upscale(image, factor)
            mock_resize.assert_called_with(image, (0, 0), fx=factor, fy=factor,
                                           interpolation=0)
