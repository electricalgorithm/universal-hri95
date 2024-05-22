"""
Unit tests for the `metrics.InterfaceMetric` class.
"""

from harmonicsradius.metrics.interface_metric import InterfaceMetric, MetricResult


class TestMetricResult:
    """Test the MetricResult class."""

    def test_constructor(self):
        """Test the constructor."""
        metric_result = MetricResult("PSNR", 10.0, "dB")
        assert metric_result.name == "PSNR"
        assert metric_result.value == 10.0
        assert metric_result.unit == "dB"

    def test_to_string(self):
        """Test the to_string method."""
        metric_result = MetricResult("PSNR", 10.0, "dB")
        metric_result.register_image_names("Baby", "Baby2")
        assert str(metric_result) == "PSNR: 10.000 dB (ref: Baby, comp: Baby2)"

    def test_to_dict(self):
        """Test the to_dict method."""
        metric_result = MetricResult("PSNR", 10.0, "dB")
        metric_result.register_image_names("Baby", "Baby2")
        assert metric_result.to_dict() == {
            "name": "PSNR",
            "value": 10.0,
            "unit": "dB",
            "referance": "Baby",
            "image": "Baby2"
        }

    def test_register_image_names(self):
        """Test the register_image_names method."""
        metric_result = MetricResult("PSNR", 10.0, "dB")
        metric_result.register_image_names("Baby", "Baby2")
        assert metric_result.reference_image_name == "Baby"
        assert metric_result.image_name == "Baby2"


class TestInterfaceMetric:
    """Test the InterfaceMetric class."""

    def test_abstracts(self):
        """Test if the class is abstract."""
        assert InterfaceMetric.__abstractmethods__ == {
            "calculate", "keywords_needed"}
