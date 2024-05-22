"""
Test the `HarmonicsRadius` metric.
"""
import pytest
from harmonicsradius.metrics.harmonics_radius import HarmonicsRadius
from harmonicsradius.image import Image


class TestHarmonicRadius:
    """Test the `HarmonicsRadius` class."""

    def test_calculate(self):
        """Test the calculate method."""
        image1 = Image("tests/data/set5_baby.png", "Baby")
        image2 = Image("tests/data/set5_baby.png", "Baby")
        hr = HarmonicsRadius()
        kwargs = {"y_true": image1, "y_pred": image2}
        result = hr.calculate(**kwargs)
        assert result.name == "harmonics_radius_95"
        assert result.value == 100.0
        assert result.unit == "%"

    def test_calculate_wrong_size(self):
        """Test the calculate method with images of different sizes."""
        image1 = Image("tests/data/set5_baby.png", "Baby")
        image2 = Image(image1.get_image()[0:-50, 0:-1], "Baby2")
        hr = HarmonicsRadius()
        kwargs = {"y_true": image1, "y_pred": image2}
        with pytest.raises(ValueError):
            hr.calculate(**kwargs)

    def test_calculate_wrong_class(self):
        """Test the calculate method with images of different classes."""
        image1 = Image("tests/data/set5_baby.png", "Baby")
        hr = HarmonicsRadius()
        kwargs = {"y_true": image1, "y_pred": "Baby"}
        with pytest.raises(TypeError):
            hr.calculate(**kwargs)

    def test_keywords_needed(self):
        """Test the keywords_needed method."""
        hr = HarmonicsRadius()
        assert "y_true" in hr.keywords_needed.keys()
        assert "y_pred" in hr.keywords_needed.keys()
