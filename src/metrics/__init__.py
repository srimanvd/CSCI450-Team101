from .base import Metric, MetricResult
from .availability import AvailabilityMetric
from .license import LicenseMetric
from .size import SizeMetric
from .dataset_code import DatasetCodePresenceMetric
from .dataset_quality import DatasetQualityMetric
from .code_quality import CodeQualityMetric
from .performance_claims import PerformanceClaimsMetric
from .bus_factor import BusFactorMetric

def metric_registry():
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
