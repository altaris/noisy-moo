"""
Subpackage for denoising algorithms. A denoiser tries to cancel noise. (also
water is wet)
* `nmoo.denoisers.knnavg.KNNAvg`
* `nmoo.denoisers.resample_average.ResampleAverage`
* `nmoo.denoisers.gpss.GPSS`
"""
__docformat__ = "google"

from .knnavg import KNNAvg
from .resample_average import ResampleAverage
from .gpss import GPSS
