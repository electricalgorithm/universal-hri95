"""
The unit-test for the `PeakSignalToNoiseRatio` class.
"""
import pytest
import numpy
from harmonicsradius.metrics.peak_signal_to_noise_ratio import PeakSignalToNoiseRatio
from harmonicsradius.image import Image


class TestPeakSignalToNoiseRatio:
    """Test the `PeakSignalToNoiseRatio` class."""

    def test_calculate(self):
        """Test the calculate method."""
        image1 = Image("tests/data/set5_baby.png", "Baby")
        image2 = Image("tests/data/set5_baby.png", "Baby")
        kwargs = {"y_true": image1, "y_pred": image2}
        psnr = PeakSignalToNoiseRatio()
        result = psnr.calculate(**kwargs)
        assert result.name == "PSNR"
        assert result.value == float("inf")
        assert result.unit == "dB"

    def test_keywords_needed(self):
        """Test the keywords_needed method."""
        psnr = PeakSignalToNoiseRatio()
        assert "y_true" in psnr.keywords_needed.keys()
        assert "y_pred" in psnr.keywords_needed.keys()

    def test_calculate_wrong_class(self):
        """Test the calculate method with wrong classes."""
        psnr = PeakSignalToNoiseRatio()
        kwargs = {"y_true": 1, "y_pred": 2}
        with pytest.raises(TypeError):
            psnr.calculate(**kwargs)

    def test_calculate_wrong_shape(self):
        """Test the calculate method with wrong shapes."""
        image1 = Image(numpy.array([[1, 2], [3, 4]]), "a")
        image2 = Image(numpy.array([[1, 2, 3], [4, 5, 6]]), "b")
        psnr = PeakSignalToNoiseRatio()
        kwargs = {"y_true": image1, "y_pred": image2}
        with pytest.raises(ValueError):
            psnr.calculate(**kwargs)
