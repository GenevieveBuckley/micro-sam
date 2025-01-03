import unittest

import torch

import micro_sam.util as util


class TestPEFTSam(unittest.TestCase):
    model_type = "vit_b"

    def _check_output(self, peft_sam):
        shape = (3, 1024, 1024)
        expected_shape = (1, 3, 1024, 1024)
        with torch.no_grad():
            batched_input = [{"image": torch.rand(*shape), "original_size": shape[1:]}]
            output = peft_sam(batched_input, multimask_output=True)
            masks = output[0]["masks"]
            self.assertEqual(masks.shape, expected_shape)

    def test_lora_sam(self):
        from micro_sam.models.peft_sam import PEFT_Sam, LoRASurgery

        _, sam = util.get_sam_model(model_type=self.model_type, return_sam=True, device="cpu")
        peft_sam = PEFT_Sam(sam, rank=2, peft_module=LoRASurgery)
        self._check_output(peft_sam)

    def test_fact_sam(self):
        from micro_sam.models.peft_sam import PEFT_Sam, FacTSurgery

        _, sam = util.get_sam_model(model_type=self.model_type, return_sam=True, device="cpu")
        peft_sam = PEFT_Sam(sam, rank=2, peft_module=FacTSurgery)
        self._check_output(peft_sam)

    def test_attention_layer_peft_sam(self):
        from micro_sam.models.peft_sam import PEFT_Sam, AttentionSurgery

        _, sam = util.get_sam_model(model_type=self.model_type, return_sam=True, device="cpu")
        peft_sam = PEFT_Sam(sam, rank=2, peft_module=AttentionSurgery)
        self._check_output(peft_sam)

    def test_norm_layer_peft_sam(self):
        from micro_sam.models.peft_sam import PEFT_Sam, LayerNormSurgery

        _, sam = util.get_sam_model(model_type=self.model_type, return_sam=True, device="cpu")
        peft_sam = PEFT_Sam(sam, rank=2, peft_module=LayerNormSurgery)
        self._check_output(peft_sam)

    def test_bias_layer_peft_sam(self):
        from micro_sam.models.peft_sam import PEFT_Sam, BiasSurgery

        _, sam = util.get_sam_model(model_type=self.model_type, return_sam=True, device="cpu")
        peft_sam = PEFT_Sam(sam, rank=2, peft_module=BiasSurgery)
        self._check_output(peft_sam)

    def test_ssf_peft_sam(self):
        from micro_sam.models.peft_sam import PEFT_Sam, SSFSurgery

        _, sam = util.get_sam_model(model_type=self.model_type, return_sam=True, device="cpu")
        peft_sam = PEFT_Sam(sam, rank=2, peft_module=SSFSurgery)
        self._check_output(peft_sam)

    def test_adaptformer_peft_sam(self):
        from micro_sam.models.peft_sam import PEFT_Sam, AdaptFormer

        _, sam = util.get_sam_model(model_type=self.model_type, return_sam=True, device="cpu")
        peft_sam = PEFT_Sam(sam, rank=2, peft_module=AdaptFormer, projection_size=64, alpha=2.0, dropout=0.5)
        self._check_output(peft_sam)


if __name__ == "__main__":
    unittest.main()
