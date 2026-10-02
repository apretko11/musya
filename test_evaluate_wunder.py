"""Data-independent checks: python -m unittest test_evaluate_wunder -v."""

import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

import evaluate_wunder as evaluation
from train_wunder import build_model
from wunder_dataset import I0_COLUMNS, I1_COLUMNS, SHARED_COLUMNS
from wunder_metric import GlobalAccumulator


class RunningSumModel(torch.nn.Module):
    def forward(self, x, hidden_states=None, return_hidden=False):
        previous = 0 if hidden_states is None else hidden_states
        prediction = x[:, :, 0, :2].cumsum(dim=1) + previous
        return prediction, prediction[:, -1:]


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_checkpoint_roundtrip_and_chunk_equivalence_all_architectures(self):
        torch.manual_seed(7)
        x = torch.randn(29, 2, 60)
        with tempfile.TemporaryDirectory() as directory:
            for name in evaluation.MODEL_NAMES:
                with self.subTest(model=name):
                    config = dict(model=name, num_blocks=2, num_iterations=2)
                    original = build_model(argparse.Namespace(**config)).eval()
                    checkpoint = Path(directory) / f"{name}.pt"
                    torch.save(original.state_dict(), checkpoint)
                    checkpoint.with_suffix(".json").write_text(json.dumps({
                        **config, "chunk_size": 7,
                    }))
                    args = evaluation.parse_args(["--checkpoint", str(checkpoint)])
                    resolved, chunk_size, _ = evaluation.resolve_config(args)
                    restored = build_model(argparse.Namespace(**resolved)).eval()
                    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
                    with torch.no_grad():
                        expected = original(x.unsqueeze(0)).squeeze(0)
                    # Repeated calls must start with fresh hidden state.
                    for _ in range(2):
                        actual = evaluation.predict_sequence(restored, x, "cpu", chunk_size)
                        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_metadata_conflicts_and_explicit_depth(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            argv = ["--checkpoint", str(checkpoint), "--model", "looped"]
            with self.assertRaisesRegex(ValueError, "num-iterations"):
                evaluation.resolve_config(evaluation.parse_args(argv))
            config, chunk_size, _ = evaluation.resolve_config(
                evaluation.parse_args(argv + ["--num-iterations", "4"])
            )
            self.assertEqual(config["num_iterations"], 4)
            self.assertEqual(chunk_size, 512)
            checkpoint.with_suffix(".json").write_text(json.dumps({
                "model": "looped", "num_iterations": 2,
            }))
            with self.assertRaisesRegex(ValueError, "conflicts"):
                evaluation.resolve_config(evaluation.parse_args(argv + ["--num-iterations", "4"]))

    def test_complete_sequences_masks_limit_and_json(self):
        rng = np.random.default_rng(9)
        length = evaluation.SEQUENCE_LENGTH
        expected = GlobalAccumulator()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            data_path = directory / "valid.parquet"
            writer = None
            try:
                for sequence in range(3):
                    columns = {
                        name: rng.normal(0, 0.01, length).astype(np.float32)
                        for name in I0_COLUMNS + I1_COLUMNS + SHARED_COLUMNS
                    }
                    need = np.arange(length) >= 1000
                    scored = np.arange(length) % 3 == 0
                    targets = rng.normal(size=(length, 2)).astype(np.float32)
                    columns.update({
                        "seq_ix": np.full(length, sequence),
                        "step_in_seq": np.arange(length),
                        "need_prediction": need,
                        "is_scored": scored,
                        "t0": targets[:, 0], "t1": targets[:, 1],
                    })
                    table = pa.table(columns)
                    if writer is None:
                        writer = pq.ParquetWriter(data_path, table.schema)
                    writer.write_table(table, row_group_size=length)
                    if sequence < 2:
                        predictions = np.cumsum(np.column_stack([
                            columns[I0_COLUMNS[0]], columns[I0_COLUMNS[1]],
                        ]).astype(np.float64), axis=0).astype(np.float32)
                        expected.add(targets, predictions, need & scored)
            finally:
                if writer is not None:
                    writer.close()

            checkpoint = directory / "model.pt"
            torch.save(RunningSumModel().state_dict(), checkpoint)
            checkpoint.with_suffix(".json").write_text(json.dumps({
                "model": "baseline", "num_blocks": 1, "chunk_size": 777,
            }))
            output = directory / "report.json"
            with patch.object(evaluation, "build_model", return_value=RunningSumModel()), \
                    patch.object(evaluation, "load_validation_sequence",
                                 wraps=evaluation.load_validation_sequence) as loader, \
                    contextlib.redirect_stdout(io.StringIO()) as stdout:
                evaluation.main([
                    "--checkpoint", str(checkpoint), "--data-path", str(data_path),
                    "--max-sequences", "2", "--device", "cpu",
                    "--output-json", str(output),
                ])
            report = json.loads(output.read_text())
            for key, value in expected.result().items():
                self.assertAlmostEqual(report[key], value, places=6)
            self.assertEqual(report["sequences"], 2)
            self.assertEqual(report["total_validation_sequences"], 3)
            self.assertFalse(report["full_validation"])
            self.assertEqual([call.args[1] for call in loader.call_args_list], [0, 1])
            self.assertIn("global weighted_pearson:", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
