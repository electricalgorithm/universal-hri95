"""
MSE implementation as a metric.
"""
import numpy
from skimage.metrics import structural_similarity
from core.image import Image
from core.metrics.interface_metric import InterfaceMetric, MetricResult


class StructuralSimilarityIndex(InterfaceMetric):
    """The SSIM metric."""

    @property
    def keywords_needed(self) -> dict[str, type]:
        """The keywords needed to calculate the metric.

        :return: The keywords needed.
        """
        return {"y_true": Image, "y_pred": Image}

    def calculate(self, **kwargs) -> MetricResult:
        """Calculate the SSIM metric.

        :param kwargs: The keywords needed to calculate the metric.
        Check the keywords_needed property.
        :return: The SSIM metric in a MetricResult object.
        """
        # Check keywords.
        if not set(self.keywords_needed).issubset(kwargs):
            raise ValueError("Missing keywords needed to calculate the metric.")

        # Get the parameters.
        y_true = kwargs["y_true"]
        y_pred = kwargs["y_pred"]

        # Check the types of the parameters.
        if not isinstance(y_true, Image):
            raise TypeError("y_true must be an Image.")
        if not isinstance(y_pred, Image):
            raise TypeError("y_pred must be an Image.")

        # Check the shapes of the parameters.
        if y_true.get_shape() != y_pred.get_shape():
            raise ValueError("y_true and y_pred must have the same shape.")

        # Calculate the SSIM.
        y_true_array: numpy.ndarray = y_true.get_image()
        y_pred_array: numpy.ndarray = y_pred.get_image()

        calculated_ssim = structural_similarity(
            y_true_array,
            y_pred_array,
            data_range=y_true_array.max() - y_true_array.min(),
            multichannel=True,
            channel_axis=2,
        )

        return MetricResult(metric_name="SSIM", metric_value=calculated_ssim, metric_unit="")
