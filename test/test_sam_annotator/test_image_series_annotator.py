import gc
import os
import platform
import tempfile

import imageio.v3 as imageio
import pytest
from skimage.data import binary_blobs
import torch.backends

import micro_sam.util as util
from micro_sam.sam_annotator import image_series_annotator, image_folder_annotator
from micro_sam.sam_annotator._state import AnnotatorState
from micro_sam._test_util import check_layer_initialization


def _create_images(tmpdir, n_images):
    image_paths = []
    for i in range(n_images):
        im_path = os.path.join(tmpdir, f"image-{i}.png")
        image_data = binary_blobs(512)
        imageio.imwrite(im_path, image_data)
        image_paths.append(im_path)
    return image_paths


@pytest.mark.gui
@pytest.mark.skipif(platform.system() == "Windows", reason="Gui test is not working on windows.")
def test_image_series_annotator(make_napari_viewer_proxy):
    """Integration test for annotator_tracking.
    """
    n_images = 3
    model_type = "vit_t" if util.VIT_T_SUPPORT else "vit_b"

    with tempfile.TemporaryDirectory() as tmpdir:
        image_paths = _create_images(tmpdir, n_images)
        output_folder = os.path.join(tmpdir, "segmentation_results")

        viewer = make_napari_viewer_proxy()
        # test generating image embedding, then adding micro-sam dock widgets to the GUI
        viewer = image_series_annotator(
            image_paths, output_folder,
            model_type=model_type,
            viewer=viewer,
            return_viewer=True,
        )

        check_layer_initialization(viewer, (512, 512))
        viewer.close()  # must close the viewer at the end of tests

    # Ensure proper garbage collection and prevent memory problems in our CI tests
    del viewer
    state = AnnotatorState()
    state.reset_state()
    del state
    gc.collect()
    # Clear pytorch cache
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.backends.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.mark.gui
@pytest.mark.skipif(platform.system() == "Windows", reason="Gui test is not working on windows.")
def test_image_folder_annotator(make_napari_viewer_proxy):
    """Integration test for annotator_tracking.
    """
    n_images = 3
    model_type = "vit_t" if util.VIT_T_SUPPORT else "vit_b"

    with tempfile.TemporaryDirectory() as tmpdir:
        _create_images(tmpdir, n_images)
        output_folder = os.path.join(tmpdir, "segmentation_results")

        viewer = make_napari_viewer_proxy()
        # test generating image embedding, then adding micro-sam dock widgets to the GUI
        viewer = image_folder_annotator(
            tmpdir, output_folder, "*.png",
            model_type=model_type,
            viewer=viewer,
            return_viewer=True,
        )

        check_layer_initialization(viewer, (512, 512))
        viewer.close()  # must close the viewer at the end of tests

    # Ensure proper garbage collection and prevent memory problems in our CI tests
    del viewer
    state = AnnotatorState()
    state.reset_state()
    del state
    gc.collect()
    # Clear pytorch cache
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.backends.cuda.is_available():
        torch.cuda.empty_cache()
