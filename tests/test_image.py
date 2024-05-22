"""
The unit-test for the  Image class.
"""
import os
import numpy
from harmonicsradius.image import Image


class TestImage:
    """The unit-test for the Image class."""

    def test_constructor_with_path(self):
        """Test the constructor with path."""
        image = Image("tests/data/set5_baby.png", "Baby")
        assert image.get_shape() == (256, 256, 3)
        assert image.get_name() == "Baby"

    def test_constructor_with_image(self):
        """Test the constructor with image."""
        image = Image("tests/data/set5_baby.png", "Baby")
        image2 = Image(image, "Baby2")
        assert image2.get_shape() == (256, 256, 3)
        assert image2.get_name() == "Baby2"

    def test_constructor_with_ndarray(self):
        """Test the constructor with ndarray."""
        image = Image("tests/data/set5_baby.png", "Baby")
        image2 = Image(image.get_image(), "Baby2")
        assert image2.get_shape() == (256, 256, 3)
        assert image2.get_name() == "Baby2"

    def test_get_image(self):
        """Test the get_image method."""
        image = Image("tests/data/set5_baby.png", "Baby")
        assert isinstance(image.get_image(), numpy.ndarray)
        assert image.get_image().shape == (256, 256, 3)

    def test_get_shape(self):
        """Test the get_shape method."""
        image = Image("tests/data/set5_baby.png", "Baby")
        assert image.get_shape() == (256, 256, 3)

    def test_get_name(self):
        """Test the get_name method."""
        image = Image("tests/data/set5_baby.png", "Baby")
        assert image.get_name() == "Baby"

    def test_save_image(self):
        """Test the save_image method."""
        image = Image("tests/data/set5_baby.png", "Baby")
        image.save_image("tests/data/set5_baby_copy.png")
        image2 = Image("tests/data/set5_baby_copy.png", "Baby")
        assert image2.get_shape() == (256, 256, 3)
        assert image2.get_name() == "Baby"
        os.remove("tests/data/set5_baby_copy.png")
