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

    def test_end_pin_frame_follows_overlap(self):
        # The pin has to land on the frame the next clip continues from, i.e. the first of the
        # saved hand-off. Step delivers frames[:-handoff], so that frame is at -handoff and the
        # last delivered one at -(handoff + 1). A hard-coded frame_idx drifts the moment
        # overlap_frames changes, and the join then cuts before the model has arrived.
        for overlap, expected in ((0, -1), (8, -9), (16, -17), (24, -25)):
            np.save(self.root/'existing/handoff_000.npy',
                    np.full((overlap + 1, 8, 8, 3), 200, np.uint8))
            r = self.state(overlap_frames=overlap)
            self.assertEqual(r[17], expected)
            self.assertEqual(r[17], -r[5]['handoff'])

        np.save(self.root/'existing/handoff_000.npy', np.full((17, 8, 8, 3), 200, np.uint8))
        for clip in (1, 2, 3):
            np.save(self.root/f'existing/handoff_{clip:03d}.npy',
                    np.full((17, 8, 8, 3), 200, np.uint8))
        with open(self.root/'existing/plan.json', 'w') as f:
            json.dump({'settings': {}, 'plan': [{'start': s*9.25, 'num_seconds': 10, 'scene': 0,
                                                 'scene_start': False, 'keep_frames': None}
                                                for s in range(4)]}, f)
        redo = self.state(overlap_frames=16, redo_session='existing', redo_clips='2')
        self.assertTrue(redo[15])            # pin_end on: clip 3 is not being regenerated
        self.assertEqual(redo[17], -17)

    def test_redo_preserves_existing_tail_pair(self):
        chain = self.state()[5] | {'redo': True, 'redo_list':[1, 3], 'redo_index':0}
        d = self.root/'existing'
        np.save(d/'handoff_001.npy', np.full((9,8,8,3), 17, np.uint8))
        np.save(d/'seam_001.npy', np.full((9,8,8,3), 19, np.uint8))
        ns['LTXChainStep']().run(torch.full((241,8,8,3), .4), chain, 'final',16,False)
        self.assertTrue(np.all(np.load(d/'handoff_001.npy') == 17))
        self.assertTrue(np.all(np.load(d/'seam_001.npy') == 19))


    def test_redo_tail_dissolves_into_the_previous_take(self):
        # The next clip is not being redone, so its head still sits on the OLD hand-off. This take
        # therefore has to end on the old frames; the end pin only bends it that way. Measured on
        # 20260923_v09: a pinned-only ending left 43-55 against 25 for a clip that got there by
        # itself, i.e. a visible step at the join.
        d = Path(self.session('existing'))
        np.save(d/'handoff_000.npy', np.full((17, 8, 8, 3), 200, np.uint8))
        np.save(d/'seam_000.npy', np.full((17, 8, 8, 3), 51, np.uint8))
        np.save(d/'handoff_001.npy', np.full((17, 8, 8, 3), 77, np.uint8))
        # the delivered tail of the take clip 3 was built on
        np.save(d/'tail_001.npy', np.full((17, 8, 8, 3), 26, np.uint8))
        (d/'clip_002.mp4').touch()          # a later attempt, NOT what clip 3 follows
        base = self.state(overlap_frames=16)[5]
        # redoing clips 2 and 4: clip 3 in between keeps its old head
        redo = base | {'redo': True, 'redo_list': [1, 3], 'redo_index': 0, 'pin_end': True}

        ns['LTXChainStep']().run(torch.full((241, 8, 8, 3), .4), redo, 'final', 16, False)
        out = self.saved[-1]['images']
        self.assertEqual(out.shape[0], 224)
        # ends on tail_001 (51/255), not on the stubbed mp4 (0.2): the reference is the take the
        # NEXT clip follows, not whichever attempt this run happens to replace
        self.assertAlmostEqual(float(out[-1].mean()), 26/255, places=5)
        self.assertTrue(abs(float(out[-1].mean()) - .2) > .002)
        # ... and the fade begins on this take's own content, not on a jump
        self.assertAlmostEqual(float(out[-17].mean()), .4, places=5)
        self.assertTrue(26/255 < float(out[-9].mean()) < .4)
        # the old hand-off and tail stay: the next clip follows them and is not being regenerated
        self.assertTrue(np.all(np.load(d/'handoff_001.npy') == 77))
        self.assertTrue(np.all(np.load(d/'tail_001.npy') == 26))

        # when the next clip IS regenerated, this take owns the join: no dissolve, and the tail it
        # delivers is recorded so a later redo of this clip alone has the right thing to fade into
        (d/'clip_002.mp4').touch()
        ns['LTXChainStep']().run(torch.full((241, 8, 8, 3), .4),
                                 redo | {'redo_list': [1, 2], 'pin_end': False}, 'final', 16, False)
        out2 = self.saved[-1]['images']
        self.assertAlmostEqual(float(out2[-1].mean()), .4, places=5)
        self.assertTrue(np.all(np.load(d/'tail_001.npy') == 102))   # 0.4 * 255
        self.assertEqual(np.load(d/'tail_001.npy').shape, (17, 8, 8, 3))

if __name__ == '__main__':
    unittest.main(verbosity=2)
