"""Signal processing algorithms."""

from quant_portfolio.algorithms.signal_processing.kalman_filter import KalmanFilterAlgorithm
from quant_portfolio.algorithms.signal_processing.wavelet_transform import WaveletTransformAlgorithm
from quant_portfolio.algorithms.signal_processing.fourier_analysis import FourierAnalysisAlgorithm
from quant_portfolio.algorithms.signal_processing.hodrick_prescott import HodrickPrescottAlgorithm
from quant_portfolio.algorithms.signal_processing.savitzky_golay import SavitzkyGolayAlgorithm
from quant_portfolio.algorithms.signal_processing.exponential_smoothing import ExponentialSmoothingAlgorithm
from quant_portfolio.algorithms.signal_processing.bandpass_filter import BandpassFilterAlgorithm
from quant_portfolio.algorithms.signal_processing.hilbert_transform import HilbertTransformAlgorithm
from quant_portfolio.algorithms.signal_processing.emd_decomposition import EMDDecompositionAlgorithm
from quant_portfolio.algorithms.signal_processing.particle_filter import ParticleFilterAlgorithm

__all__ = [
    "KalmanFilterAlgorithm",
    "WaveletTransformAlgorithm",
    "FourierAnalysisAlgorithm",
    "HodrickPrescottAlgorithm",
    "SavitzkyGolayAlgorithm",
    "ExponentialSmoothingAlgorithm",
    "BandpassFilterAlgorithm",
    "HilbertTransformAlgorithm",
    "EMDDecompositionAlgorithm",
    "ParticleFilterAlgorithm",
]
