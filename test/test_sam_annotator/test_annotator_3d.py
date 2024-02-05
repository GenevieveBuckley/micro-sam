import gc
import platform

import numpy as np
import pytest
from skimage.data import binary_blobs
import torch

import micro_sam.util as util
from micro_sam._test_util import check_layer_initialization
from micro_sam.sam_annotator import annotator_3d
from micro_sam.sam_annotator._state import AnnotatorState


@pytest.mark.gui
@pytest.mark.skipif(platform.system() == "Windows", reason="Gui test is not working on windows.")
def test_annotator_3d(make_napari_viewer_proxy):
    """Integration test for annotator_3d.
    """

    image = np.stack(4 * [binary_blobs(512)])
    model_type = "vit_t" if util.VIT_T_SUPPORT else "vit_b"

    viewer = make_napari_viewer_proxy()
    # test generating image embedding, then adding micro-sam dock widgets to the GUI
    viewer = annotator_3d(
        image,
        model_type=model_type,
        viewer=viewer,
        return_viewer=True
    )

    check_layer_initialization(viewer, image.shape)
    viewer.close()  # must close the viewer at the end of tests

    # Ensure proper garbage collection and prevent memory problems in our CI tests
    del viewer
    del image
    state = AnnotatorState()
    state.reset_state()
    del state
    gc.collect()
    # Clear pytorch cache
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
