import numpy as np

from micro_sam.sam_annotator import annotator_2d
from micro_sam.sam_annotator.annotator_2d import _initialize_viewer


def test_open_annotator_2d(make_napari_viewer_proxy, tmp_path):
    """Integration test for annotator_2d widget.

    Creates a 2D image embedding and opens annotator_2d dock widgets in napari.
    Does not test user interaction with micro-sam widgets.
    """
    model_type = "vit_t"  # tiny model
    embedding_path = tmp_path / "test-embedding.zarr"
    # example data - a basic checkerboard pattern
    image = np.zeros((16, 16))
    image[:8, :8] = 1
    image[8:, 8:] = 1

    viewer = make_napari_viewer_proxy()
    viewer = _initialize_viewer(image, None, None, None)
    # Test generating image embedding then add micro-sam dock widgets to the GUI
    viewer = annotator_2d(
        image,
        embedding_path,
        show_embeddings=False,
        model_type=model_type,
        v=viewer,
        return_viewer=True,
    )
    assert len(viewer.layers) == 6
    expected_layer_names = [
        "raw",
        "auto_segmentation",
        "committed_objects",
        "current_object",
        "point_prompts",
        "prompts",
    ]
    for layername in expected_layer_names:
        assert layername in viewer.layers
    # The annotator 2d dock widgets are now open in the napari viewer.
    viewer.close()  # must close the viewer at the end of tests
