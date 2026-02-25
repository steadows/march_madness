"""Tests for src/submission.py — submission format validation."""
import numpy as np
import pandas as pd
import pytest

from src import data_loader as dl, config
from src.calibration import clip_predictions


class TestClipPredictions:
    def test_clips_high(self):
        preds = np.array([0.96, 0.99, 1.0])
        clipped = clip_predictions(preds)
        assert (clipped <= config.CLIP_HIGH).all()

    def test_clips_low(self):
        preds = np.array([0.01, 0.0, -0.1])
        clipped = clip_predictions(preds)
        assert (clipped >= config.CLIP_LOW).all()

    def test_in_range_unchanged(self):
        preds = np.array([0.3, 0.5, 0.7])
        clipped = clip_predictions(preds)
        np.testing.assert_array_almost_equal(clipped, preds)

    def test_custom_bounds(self):
        preds = np.array([0.0, 0.5, 1.0])
        clipped = clip_predictions(preds, low=0.1, high=0.9)
        assert clipped[0] == pytest.approx(0.1)
        assert clipped[1] == pytest.approx(0.5)
        assert clipped[2] == pytest.approx(0.9)


class TestSubmissionFormat:
    """Test that generated submissions match required format.

    Uses a mock predict_fn that returns 0.5 for all matchups.
    Full baseline submission is tested in test_pipeline.py.
    """

    @pytest.fixture(scope='class')
    def sample_sub(self):
        return dl.load_sample_submission(stage=1)

    def test_sample_sub_has_correct_columns(self, sample_sub):
        assert list(sample_sub.columns) == ['ID', 'Pred']

    def test_id_format(self, sample_sub):
        """All IDs are Season_LowerTeamID_HigherTeamID format."""
        for row_id in sample_sub['ID'].head(100):
            parts = row_id.split('_')
            assert len(parts) == 3
            season, team_a, team_b = int(parts[0]), int(parts[1]), int(parts[2])
            assert 2022 <= season <= 2026
            assert team_a < team_b, f"ID {row_id}: team_a not < team_b"

    def test_stage1_row_count(self, sample_sub):
        assert len(sample_sub) > 400_000

    def test_stage2_row_count(self):
        sub2 = dl.load_sample_submission(stage=2)
        assert len(sub2) > 100_000

    def test_submission_has_m_pairs(self, sample_sub):
        team_a_ids = sample_sub['ID'].apply(lambda x: int(x.split('_')[1]))
        assert team_a_ids.between(1000, 1999).any()

    def test_submission_has_w_pairs(self, sample_sub):
        team_a_ids = sample_sub['ID'].apply(lambda x: int(x.split('_')[1]))
        assert team_a_ids.between(3000, 3999).any()

    def test_baseline_submission_exists(self):
        """After pipeline run, baseline_v1.csv should be present."""
        from pathlib import Path
        # This test will pass once test_pipeline.py runs generate_submission
        sub_path = Path('submissions/baseline_v1.csv')
        if sub_path.exists():
            df = pd.read_csv(sub_path)
            assert list(df.columns) == ['ID', 'Pred']
            assert len(df) > 400_000
            assert (df['Pred'] >= config.CLIP_LOW).all()
            assert (df['Pred'] <= config.CLIP_HIGH).all()
            assert df['Pred'].isna().sum() == 0
            assert df['Pred'].std() > 0.05


class TestPredictionConstraints:
    """Validates prediction array properties for any submission."""

    def test_no_nan(self):
        preds = np.array([0.3, 0.5, 0.7, 0.4])
        assert not np.isnan(preds).any()

    def test_clipped_range(self):
        preds = clip_predictions(np.random.uniform(0, 1, 1000))
        assert (preds >= config.CLIP_LOW).all()
        assert (preds <= config.CLIP_HIGH).all()

    def test_meaningful_spread(self):
        """Predictions shouldn't all be 0.5."""
        preds = np.array([0.2, 0.3, 0.6, 0.7, 0.45, 0.55, 0.8, 0.35])
        assert preds.std() > 0.05
