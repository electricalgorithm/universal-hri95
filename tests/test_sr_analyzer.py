"""
The unit-test for the SRAnalyzer class.
"""
import pytest
import numpy
from harmonicsradius.metrics import HarmonicsRadius
from harmonicsradius.image import Image
from harmonicsradius.settings import SRAnalyzerSettings
from harmonicsradius.sr_analyzer import SRAnalyzer


class TestSRAnalyzer:
    """The unit-test for the SRAnalyzer class."""

    @pytest.fixture
    def fake_image(self):
        """Fake image for testing."""
        return numpy.zeros((100, 100, 3), numpy.uint8)

    def test_constructor_without_parameters(self):
        """Test the constructor without parameters."""
        analyzer = SRAnalyzer()
        assert analyzer._settings == SRAnalyzerSettings()
        assert analyzer._metrics == []
        assert analyzer._reference is None
        assert analyzer._images == []
        assert not analyzer._is_done

    def test_constructor_with_parameters(self):
        """Test the constructor with parameters."""
        settings = SRAnalyzerSettings()
        analyzer = SRAnalyzer(settings)
        assert analyzer._settings == settings
        assert analyzer._metrics == []
        assert analyzer._reference is None
        assert analyzer._images == []
        assert not analyzer._is_done

    def test_add_metric(self):
        """Test the add_metric method."""
        analyzer = SRAnalyzer()
        metric = HarmonicsRadius()
        analyzer.add_metric(metric)
        assert analyzer._metrics == [metric]

    def test_add_reference_image(self, fake_image):
        """Test the add_reference_image method."""
        analyzer = SRAnalyzer()
        image = Image(fake_image, "x")
        analyzer.add_reference_image(image)
        assert analyzer._reference == image

    def test_add_image(self, fake_image):
        """Test the add_image method."""
        analyzer = SRAnalyzer()
        image = Image(fake_image, "x")
        analyzer.add_image(image)
        assert analyzer._images == [image]

    def test_calculate(self, fake_image):
        """Test the calculate method."""
        analyzer = SRAnalyzer()
        analyzer.add_metric(HarmonicsRadius())
        with pytest.raises(RuntimeError) as excinfo:
            analyzer.calculate()
        assert str(excinfo.value) == "The reference image is not set."

        image = Image(fake_image, "image_name")
        analyzer.add_reference_image(image)
        with pytest.raises(RuntimeError) as excinfo:
            analyzer.calculate()
        assert str(excinfo.value) == "There is no image to calculate."

        image = Image(fake_image, "image_name")
        analyzer.add_image(image)
        analyzer.calculate()
        with pytest.raises(RuntimeError) as excinfo:
            analyzer.calculate()
        assert str(excinfo.value) == "The analyzer is already done."

    def test_calculate_keywords(self, fake_image):
        """Test the calculate method with keywords."""
        analyzer = SRAnalyzer()
        analyzer.add_metric(HarmonicsRadius())
        image = Image(fake_image, "image_name")
        analyzer.add_reference_image(image)
        image = Image(fake_image, "image_name")
        analyzer.add_image(image)
        results = analyzer.calculate()
        assert len(results) == 1
        assert results[0].name == "harmonics_radius_95"
        assert results[0].value == 0.0
        assert results[0].unit == "%"
        assert analyzer._is_done
