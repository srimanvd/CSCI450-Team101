from .availability import AvailabilityMetric
from .base import Metric, MetricResult
from .bus_factor import BusFactorMetric
from .code_quality import CodeQualityMetric
from .dataset_code import DatasetCodePresenceMetric
from .dataset_quality import DatasetQualityMetric
from .license import LicenseMetric
from .performance_claims import PerformanceClaimsMetric
from .size import SizeMetric

__all__ = ["Metric", "MetricResult", "metric_registry"]


def metric_registry() -> list[Metric]:
    return [
        AvailabilityMetric(),
        LicenseMetric(),
        SizeMetric(),
        DatasetCodePresenceMetric(),
        DatasetQualityMetric(),
        CodeQualityMetric(),
        PerformanceClaimsMetric(),
        BusFactorMetric(),
    ]
