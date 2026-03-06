import PupilProcessing
import PupilSense
import xdetectioncore
import numpy as np


def test_numpy_version():
    """Ensure we aren't using the broken NumPy 2.0."""
    major_version = int(np.__version__.split('.')[0])
    assert major_version < 2, f"NumPy version {np.__version__} is incompatible with Matplotlib!"

def test_package_import():
    """Verify the package is installed and has a valid file path."""
    assert xdetectioncore.__file__ is not None
    assert "site-packages" in xdetectioncore.__file__ or "XdetectionCore" in xdetectioncore.__file__
    assert PupilProcessing.__file__ is not None
    assert "site-packages" in PupilProcessing.__file__ or "PupilProcessing" in PupilProcessing.__file__
    assert PupilSense.__file__ is not None
    assert "site-packages" in PupilSense.__file__ or "PupilSense" in PupilSense.__file__

def test_inference_import():
    """Verify the Inference class can be imported."""
    from PupilSense.inference_pupil_sense import Inference
    assert Inference is not None, "Failed to import Inference class from PupilSense.inference_pupil_sense"

def test_extract_pupil_import():
    """Verify the main function can be imported."""
    from PupilSense.extract_pupil_pupil_sense import main
    assert main is not None, "Failed to import main function from PupilSense.extract_pupil_pupil_sense"

def test_path_conversion():
    """Verify the Windows-to-POSIX engine works."""
    from xdetectioncore.paths import posix_from_win
    win_path = r"X:\Data\test"
    # Adjust the expected output based on your specific lab mapping
    posix_path = posix_from_win(win_path)
    assert posix_path.root == ''
    assert str(posix_path).startswith('Data'), f"Path conversion failed: {posix_path}"