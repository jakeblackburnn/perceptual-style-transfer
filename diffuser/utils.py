import csv
import os

from style_transfer.utils.metrics import MetricsLogger, ensure_dir


class DiffusionMetricsLogger(MetricsLogger):
    """MetricsLogger variant that logs mse_loss/perceptual_loss separately.

    The feed-forward model's MetricsLogger.save() hardcodes a single 'loss'
    column, which would hide exactly the pathology observed in the failing
    notebook (noise-prediction MSE trending up while the combined loss
    looked merely noisy). Splitting the two out makes that failure mode
    visible in metrics.csv rather than only in a debugger.
    """

    def save(self):
        if not self.rows:
            return
        ensure_dir(os.path.dirname(self.logfile))
        ordered_fields = ['stage', 'epoch', 'mse_loss', 'perceptual_loss', 'loss', 'time']
        with open(self.logfile, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=ordered_fields)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)
