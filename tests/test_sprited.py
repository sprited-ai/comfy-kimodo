"""Run with ComfyUI on PYTHONPATH and the plugin dependencies installed."""
import unittest
from types import SimpleNamespace

import numpy as np
import torch
import nodes
from kimodo.skeleton.definitions import SOMASkeleton30
from kimodo.model.load_model import _build_local_text_encoder_conf
from kimodo.model.llm2vec.llm2vec import LLM2Vec


class SpritedTests(unittest.TestCase):
    def test_expanded_motion_postprocess(self):
        skeleton = SOMASkeleton30()
        rotations = torch.eye(3).repeat(1, 10, 30, 1, 1)
        roots = torch.zeros(1, 10, 3)
        _, joints, _ = skeleton.fk(rotations, roots)
        output = skeleton.output_to_SOMASkeleton77(dict(
            local_rot_mats=rotations, root_positions=roots, posed_joints=joints,
            foot_contacts=torch.zeros(1, 10, 4),
        ))
        output = {key: value.numpy() for key, value in output.items()}
        model = SimpleNamespace(skeleton=skeleton, fps=30)
        motion = nodes.Kimodo_Sampler._wrap_output(model, output, ['standing'], [10], 1, [])
        self.assertIs(motion.skeleton, skeleton.somaskel77)
        corrected, = nodes.Kimodo_PostProcess().process(motion)
        self.assertIsNot(corrected, motion, 'PostProcess silently returned uncorrected input')
        self.assertEqual(corrected.output_dict['posed_joints'].shape, (1, 10, 77, 3))
        self.assertTrue(np.isfinite(corrected.output_dict['posed_joints']).all())

    def test_encoder_selection_does_not_mutate_default(self):
        default = _build_local_text_encoder_conf()
        merged = 'raducius/Llama-3-8B-Instruct-LLM2Vec-mntp-merged'
        selected = _build_local_text_encoder_conf(merged)
        self.assertEqual(selected['base_model_name_or_path'], merged)
        self.assertEqual(selected['peft_model_name_or_path'], default['peft_model_name_or_path'])
        self.assertEqual(_build_local_text_encoder_conf(), default)

    def test_merged_instruction_format(self):
        def prepare(name):
            encoder = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(_name_or_path=name)))
            return LLM2Vec.prepare_for_tokenization(encoder, ' walking ')
        self.assertEqual(prepare('raducius/Llama-3-8B-Instruct-LLM2Vec-mntp-merged'),
                         prepare('meta-llama/Meta-Llama-3-8B-Instruct'))
        self.assertIn('<|start_header_id|>user', prepare('raducius/Llama-3-8B-Instruct-LLM2Vec-mntp-merged'))


if __name__ == '__main__':
    unittest.main()
