"""
Test the `StructuralSimilarityIndex` class.
"""

import pytest
from harmonicsradius.metrics.structural_similarity_index import StructuralSimilarityIndex
from harmonicsradius.image import Image


class TestStructuralSimilarityIndex:
    """Test the `StructuralSimilarityIndex` class."""

    def test_calculate(self):
        """Test the calculate method."""
        image1 = Image("tests/data/set5_baby.png", "Baby")
        image2 = Image("tests/data/set5_baby.png", "Baby")
        ssi = StructuralSimilarityIndex()
        kwargs = {"y_true": image1, "y_pred": image2}
        result = ssi.calculate(**kwargs)
        assert result.name == "SSIM"
        assert result.value == 1.0
        assert result.unit == ""

    def test_calculate_wrong_size(self):
        """Test the calculate method with images of different sizes."""
        image1 = Image("tests/data/set5_baby.png", "Baby")
        image2 = Image(image1.get_image()[0:-50, 0:-1], "Baby2")
        ssi = StructuralSimilarityIndex()
        kwargs = {"y_true": image1, "y_pred": image2}
        with pytest.raises(ValueError):
            ssi.calculate(**kwargs)

    def test_calculate_wrong_class(self):
        """Test the calculate method with images of different classes."""
        image1 = Image("tests/data/set5_baby.png", "Baby")
        ssi = StructuralSimilarityIndex()
        kwargs = {"y_true": image1, "y_pred": "Baby"}
        with pytest.raises(TypeError):
            ssi.calculate(**kwargs)

    def test_keywords_needed(self):
        """Test the keywords_needed method."""
        ssi = StructuralSimilarityIndex()
        assert "y_true" in ssi.keywords_needed.keys()
        assert "y_pred" in ssi.keywords_needed.keys()
