import numpy as np


METRIC_CLIP = 2.0


class GlobalAccumulator:
    """
    Streaming implementation of the Wunder
    Global Weighted Pearson metric.

    Accumulates statistics across complete sequences
    without storing the entire validation dataset.
    """

    def __init__(self):
        self.blocks_seen = 0
        self.selected_rows = 0
        self.nonempty_blocks = 0

        self.weight = np.zeros(
            2,
            dtype=np.float64,
        )

        self.mean_y = np.zeros(
            2,
            dtype=np.float64,
        )

        self.mean_p = np.zeros(
            2,
            dtype=np.float64,
        )

        self.m2_y = np.zeros(
            2,
            dtype=np.float64,
        )

        self.m2_p = np.zeros(
            2,
            dtype=np.float64,
        )

        self.cross = np.zeros(
            2,
            dtype=np.float64,
        )

    def add(
        self,
        targets,
        predictions,
        mask,
    ):
        targets = np.asarray(
            targets,
            dtype=np.float32,
        )

        predictions = np.asarray(
            predictions,
            dtype=np.float32,
        )

        mask = np.asarray(
            mask,
            dtype=bool,
        )

        if targets.shape != predictions.shape:
            raise ValueError(
                "targets and predictions must "
                "have identical shapes"
            )

        if (
            targets.ndim != 2
            or targets.shape[1] != 2
        ):
            raise ValueError(
                "expected [T, 2] targets"
            )

        if mask.shape != (
            targets.shape[0],
        ):
            raise ValueError(
                "mask has incorrect shape"
            )

        if not np.isfinite(
            targets
        ).all():
            raise ValueError(
                "targets contain nonfinite values"
            )

        if not np.isfinite(
            predictions
        ).all():
            raise ValueError(
                "predictions contain nonfinite values"
            )

        self.blocks_seen += 1

        self.selected_rows += int(
            mask.sum()
        )

        if not mask.any():
            return

        self.nonempty_blocks += 1

        y = np.clip(
            targets[mask],
            -METRIC_CLIP,
            METRIC_CLIP,
        ).astype(np.float64)

        p = np.clip(
            predictions[mask],
            -METRIC_CLIP,
            METRIC_CLIP,
        ).astype(np.float64)

        for side in range(2):
            w = np.abs(
                y[:, side]
            )

            weight = w.sum()

            if weight == 0:
                continue

            mean_y = (
                np.sum(
                    w * y[:, side]
                )
                / weight
            )

            mean_p = (
                np.sum(
                    w * p[:, side]
                )
                / weight
            )

            yc = (
                y[:, side]
                - mean_y
            )

            pc = (
                p[:, side]
                - mean_p
            )

            total = (
                self.weight[side]
                + weight
            )

            delta_y = (
                mean_y
                - self.mean_y[side]
            )

            delta_p = (
                mean_p
                - self.mean_p[side]
            )

            correction = (
                self.weight[side]
                * weight
                / total
            )

            self.m2_y[side] += (
                np.sum(
                    w * yc * yc
                )
                + delta_y**2
                * correction
            )

            self.m2_p[side] += (
                np.sum(
                    w * pc * pc
                )
                + delta_p**2
                * correction
            )

            self.cross[side] += (
                np.sum(
                    w * yc * pc
                )
                + delta_y
                * delta_p
                * correction
            )

            self.mean_y[side] += (
                delta_y
                * weight
                / total
            )

            self.mean_p[side] += (
                delta_p
                * weight
                / total
            )

            self.weight[side] = total

    def result(self):
        if not self.selected_rows:
            raise ValueError(
                "no rows selected"
            )

        per_target = np.zeros(
            2,
            dtype=np.float64,
        )

        for side in range(2):
            total = self.weight[side]

            if total < 1e-8:
                continue

            std_y = np.sqrt(
                max(
                    0.0,
                    self.m2_y[side]
                    / total,
                )
            )

            std_p = np.sqrt(
                max(
                    0.0,
                    self.m2_p[side]
                    / total,
                )
            )

            if (
                std_y > 1e-8
                and std_p > 1e-8
            ):
                correlation = (
                    self.cross[side]
                    / total
                    / (std_y * std_p)
                )

                per_target[side] = np.clip(
                    correlation,
                    -1.0,
                    1.0,
                )

        return {
            "t0": float(
                per_target[0]
            ),
            "t1": float(
                per_target[1]
            ),
            "weighted_pearson": float(
                per_target.mean()
            ),
            "blocks": (
                self.blocks_seen
            ),
            "blocks_with_scored_rows": (
                self.nonempty_blocks
            ),
            "selected_rows": (
                self.selected_rows
            ),
        }
