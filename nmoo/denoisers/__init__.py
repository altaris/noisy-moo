"""
Subpackage for denoising algorithms. A denoiser tries to cancel noise. (also
water is wet)
* `nmoo.denoisers.knnavg.KNNAvg`
* `nmoo.denoisers.resample_average.ResampleAverage`
"""
__docformat__ = "google"

from .resample_average import ResampleAverage
from .knnavg import KNNAvg
