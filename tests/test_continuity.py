"""Exercise State/Step with real tensors; stub only ComfyUI and the encoder."""
import ast
import json
import logging
import math
import os
from pathlib import Path
import re
import tempfile
from fractions import Fraction
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from PIL import Image

SOURCE = Path(__file__).resolve().parents[1] / 'ComfyUI-LTX-Chain/nodes.py'
tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
names = {'_audio_seconds', '_compatible_lead_frames', '_parse_cuts', '_schedule',
         'LTXChainState', 'LTXChainStep'}
ns = dict(globals(), CHAIN_TYPE='LTX_CHAIN', STATE_NODE_CLASS='LTXChainState',
          RES_PRESETS=['test'], CHAINS_SUBDIR='test', CHUNK_CRF=12, MIN_TAIL_SECONDS=1.)
exec(compile(ast.Module(body=[n for n in tree.body if getattr(n, 'name', '') in names],
                        type_ignores=[]), str(SOURCE), 'exec'), ns)


class Continuity(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.saved = []
        def session(name):
            p = self.root / name
            p.mkdir(exist_ok=True)
            return str(p)
        self.session = session
        ns.update(_session_dir=session, _chains_root=lambda: str(self.root),
                  _new_session_name=lambda: 'new', _parse_resolution=lambda r: (8, 8),
                  _fit_image=lambda x, w, h: x,
                  _video_tail=lambda p, k: torch.full((min(k, 232), 8, 8, 3), .2))
        self.audio = {'waveform': torch.arange(24000).view(1, 1, -1), 'sample_rate': 240}
        self.kw = dict(image=torch.zeros(1, 8, 8, 3), audio=self.audio,
                       audio_start_sec=0, chunk_seconds=10, length_mode='seconds',
                       length_seconds=30, fps=24, chain_iter=1, handoff_color_match=0,
                       chain_state=json.dumps({'session': 'existing',
                           'prev_frames': str(self.root/'existing/handoff_000.npy')}))
        d = Path(session('existing'))
        np.save(d/'handoff_000.npy', np.full((9, 8, 8, 3), 200, np.uint8))
        np.save(d/'seam_000.npy', np.full((9, 8, 8, 3), 51, np.uint8))
        (d/'clip_001.mp4').touch()
        owner = self
        class Video:
            def __init__(self, c): self.c = c
            def save_to(self, path, **kwargs): owner.saved.append(self.c)
        ns.update(InputImpl=SimpleNamespace(VideoFromComponents=Video),
                  Types=SimpleNamespace(VideoComponents=lambda **k: k,
                    VideoContainer=SimpleNamespace(MP4='mp4'), VideoCodec=SimpleNamespace(H264='h264')))

    def state(self, **kwargs):
        return ns['LTXChainState']().run(**(self.kw | kwargs))

    def test_normal_lead_and_separate_seam(self):
        r = self.state(continuity_lead_seconds=1)
        self.assertEqual(r[0].shape[0], 33)
        self.assertEqual(r[2], 11)
        self.assertAlmostEqual(r[1], 9 + 2/3 - 1)
        self.assertTrue(r[5]['prev_handoff'].endswith('seam_000.npy'))
        self.assertAlmostEqual(float(r[0][-1].mean()), 200/255, places=5)
        self.assertEqual(r[5]['lead_frames'], 24)

    def test_default_legacy_and_scene(self):
        self.assertEqual(self.state()[0].shape[0], 9)
        (self.root/'existing/seam_000.npy').unlink()
        self.assertTrue(self.state()[5]['prev_handoff'].endswith('handoff_000.npy'))
        r = self.state(continuity_lead_seconds=1, clips_per_scene=1, scene_crossfade=False)
        self.assertEqual(r[5]['lead_frames'], 0)
        self.assertIsNone(r[5]['prev_handoff'])
        r = self.state(continuity_lead_seconds=1, chain_iter=0, chain_state='')
        self.assertEqual(r[5]['lead_frames'], 0)

    def test_integer_seconds_and_latent_grid(self):
        fn = ns['_compatible_lead_frames']
        for available, fps, expected in [(24,24,24),(23,24,0),(47,24,24),
                                          (30,30,0),(120,30,120),(75,25,0)]:
            self.assertEqual(fn(available, fps, 100), expected)
        self.assertEqual(fn(24,24,.5), 0)

    def test_step_trims_audio_and_preserves_frame_count(self):
        chain = self.state(continuity_lead_seconds=1)[5]
        frames = torch.full((265,8,8,3), .4)
        ns['LTXChainStep']().run(frames, chain, 'final', 16, False,
            chunk_audio=self.audio, handoff_images=torch.full((9,8,8,3), .8))
        result = self.saved[-1]
        self.assertEqual(result['images'].shape[0], 232)
        self.assertEqual(result['audio']['waveform'][0,0,0], 240)
        self.assertEqual(self.audio['waveform'][0,0,0], 0)
        self.assertAlmostEqual(float(result['images'][0].mean()), .2, places=5)
        d = self.root/'existing'
        self.assertTrue(np.all(np.load(d/'handoff_001.npy') == 204))
        self.assertTrue(np.all(np.load(d/'seam_001.npy') == 102))

    def test_redo_preserves_existing_tail_pair(self):
        chain = self.state()[5] | {'redo': True, 'redo_list':[1, 3], 'redo_index':0}
        d = self.root/'existing'
        np.save(d/'handoff_001.npy', np.full((9,8,8,3), 17, np.uint8))
        np.save(d/'seam_001.npy', np.full((9,8,8,3), 19, np.uint8))
        ns['LTXChainStep']().run(torch.full((241,8,8,3), .4), chain, 'final',16,False)
        self.assertTrue(np.all(np.load(d/'handoff_001.npy') == 17))
        self.assertTrue(np.all(np.load(d/'seam_001.npy') == 19))


if __name__ == '__main__':
    unittest.main(verbosity=2)
