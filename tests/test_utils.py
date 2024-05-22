"""
Tests for the harmonicsradius.utils module.
"""
from unittest.mock import MagicMock, patch
from harmonicsradius.utils import (
    read_image,
    show_image,
    save_image,
)


def test_read_image():
    """Test the read_image function."""
    with patch("harmonicsradius.utils.cv2", autospec=True) as cv2:
        cv2.imread.return_value = "image"
        assert read_image("image_path") == "image"
        cv2.imread.assert_called_once_with("image_path")


def test_show_image():
    """Test the show_image function."""
    with patch("harmonicsradius.utils.cv2", autospec=True) as cv2:
        cv2.imshow = MagicMock()
        cv2.waitKey = MagicMock()
        cv2.destroyAllWindows = MagicMock()

        show_image("image")
        cv2.imshow.assert_called_once_with("Image", "image")
        cv2.waitKey.assert_called_once_with(0)
        cv2.destroyAllWindows.assert_called_once()


def test_save_image():
    """Test the save_image function."""
    with patch("harmonicsradius.utils.cv2", autospec=True) as cv2:
        cv2.imwrite = MagicMock()

        save_image("image", "image_path")
        cv2.imwrite.assert_called_once_with("image_path", "image")


def test_get_fft_of_image_color_image_without_scale_log():
    """Test the get_fft_of_image function."""
