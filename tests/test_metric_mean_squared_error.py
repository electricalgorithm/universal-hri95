"""
Test the `mean_squared_error` metric.
"""

import pytest
from harmonicsradius.metrics.mean_squared_error import MeanSquaredError
from harmonicsradius.image import Image


class TestMeanSquaredError:
    """Test the `MeanSquaredError` class."""

    def test_calculate(self):
        """Test the calculate method."""
        image1 = Image("tests/data/set5_baby.png", "Baby")
        image2 = Image("tests/data/set5_baby.png", "Baby")
        mse = MeanSquaredError()
        kwargs = {"y_true": image1, "y_pred": image2}
        result = mse.calculate(**kwargs)
        assert result.name == "mse"
        assert result.value == 0.0
        assert result.unit == "px^2"

    def test_calculate_wrong_size(self):
        """Test the calculate method with images of different sizes."""
        image1 = Image("tests/data/set5_baby.png", "Baby")
        image2 = Image(image1.get_image()[0:-50, 0:-1], "Baby2")
        mse = MeanSquaredError()
        kwargs = {"y_true": image1, "y_pred": image2}
        with pytest.raises(ValueError):
            mse.calculate(**kwargs)

    def test_calculate_wrong_class(self):
        """Test the calculate method with images of different classes."""
        image1 = Image("tests/data/set5_baby.png", "Baby")
        mse = MeanSquaredError()
        kwargs = {"y_true": image1, "y_pred": "Baby"}
        with pytest.raises(TypeError):
            mse.calculate(**kwargs)

    def test_keywords_needed(self):
        """Test the keywords_needed method."""
        mse = MeanSquaredError()
        assert "y_true" in mse.keywords_needed.keys()
        assert "y_pred" in mse.keywords_needed.keys()
