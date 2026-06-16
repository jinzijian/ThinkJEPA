import unittest
from tempfile import TemporaryDirectory

import torch

from cache_train.omnijepa import (
    MODE_ACT,
    OmniJepaBridge,
    OmniJepaCacheRecord,
    OmniJepaConfig,
    normalize_omnijepa_mode,
    route_omnijepa_output,
)
from cache_train.omnijepa_data import OmniJepaNpzDataset, write_omnijepa_cache_npz


class OmniJepaSmokeTest(unittest.TestCase):
    def test_mode_router(self):
        self.assertEqual(normalize_omnijepa_mode("<MODE=QA>"), "qa")
        self.assertEqual(normalize_omnijepa_mode("mode=act"), MODE_ACT)
        self.assertEqual(route_omnijepa_output("<MODE=PLAN>"), "text")
        self.assertEqual(route_omnijepa_output("<MODE=ACT>"), "action")

    def test_cache_record_validation(self):
        record = OmniJepaCacheRecord.from_mapping(
            {
                "episode_id": "ep0",
                "timestep": 3,
                "mode": "<MODE=ACT>",
                "instruction": "move the cube",
                "obs_frames": "obs.npy",
                "current_jepa_latent": "cur.npy",
                "predicted_future_jepa_latent": "pred.npy",
                "oracle_future_jepa_latent": "oracle.npy",
                "target_action_chunk": "action.npy",
            }
        )
        self.assertEqual(record.mode, "act")

    def test_bridge_shapes_and_action_loss(self):
        torch.manual_seed(7)
        cfg = OmniJepaConfig(
            vlm_dim=64,
            jepa_dim=32,
            action_dim=7,
            action_horizon=4,
            future_token_count=5,
            guidance_layers=3,
            guidance_tokens=4,
            guidance_dim=48,
            hidden_dim=64,
            num_heads=4,
            dropout=0.0,
        )
        bridge = OmniJepaBridge(cfg)
        B, N, T, P = 2, 9, 3, 6
        mid_hidden = torch.randn(B, N, cfg.vlm_dim)
        late_hidden = torch.randn(B, N, cfg.vlm_dim)
        future_latent = torch.randn(B, T, P, cfg.jepa_dim)
        actions = torch.randn(B, cfg.action_horizon, cfg.action_dim)

        guidance = bridge.build_jepa_guidance(mid_hidden)
        self.assertEqual(guidance["vlm_old"].shape, (B, cfg.guidance_layers, cfg.guidance_tokens, cfg.guidance_dim))
        self.assertEqual(guidance["vlm_new_mask"].shape, (B, cfg.guidance_layers, cfg.guidance_tokens))

        future_tokens = bridge.build_future_tokens(future_latent)
        self.assertEqual(future_tokens.shape, (B, cfg.future_token_count, cfg.vlm_dim))

        fused_hidden = bridge.fuse_text_hidden(late_hidden, future_latent)
        self.assertEqual(fused_hidden.shape, late_hidden.shape)

        loss, metrics = bridge.action_loss(actions, late_hidden, future_latent)
        self.assertEqual(tuple(loss.shape), ())
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("flow_loss", metrics)
        self.assertIn("action_l1", metrics)

    def test_npz_cache_roundtrip(self):
        with TemporaryDirectory() as td:
            path = f"{td}/sample.npz"
            write_omnijepa_cache_npz(
                path,
                episode_id="ep1",
                timestep=5,
                mode="<MODE=ACT>",
                instruction="push the block",
                obs_frames=torch.zeros(2, 8, 8, 3).numpy(),
                current_jepa_latent=torch.zeros(1, 2, 3).numpy(),
                predicted_future_jepa_latent=torch.ones(2, 3, 4, 5).numpy(),
                oracle_future_jepa_latent=torch.zeros(2, 3, 4, 5).numpy(),
                target_action_chunk=torch.randn(4, 7).numpy(),
            )
            ds = OmniJepaNpzDataset([path], include_obs_frames=True)
            sample = ds[0]
            self.assertEqual(sample["mode"], "act")
            self.assertEqual(sample["instruction"], "push the block")
            self.assertEqual(sample["predicted_future_jepa_latent"].shape, (2, 3, 4, 5))
            self.assertEqual(sample["target_action_chunk"].shape, (4, 7))
            self.assertIn("obs_frames", sample)


if __name__ == "__main__":
    unittest.main()
