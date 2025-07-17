import os
import csv
import torch

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

class MetricsLogger:   # Logs training metrics and saves them to a CSV file.

    def __init__(self, logfile): 
        self.logfile = logfile
        self.fields = None
        self.rows = []

    def log(self, data: dict): 
        if self.fields is None:
            self.fields = list(data.keys())
        self.rows.append(data)

    def save(self):
        if not self.rows:
            return
        ensure_dir(os.path.dirname(self.logfile))
        with open(self.logfile, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)


def save_checkpoint(model, optimizer, epoch, stage_idx, out_dir, prefix='ckpt'):

    ensure_dir(out_dir)
    checkpoint = {
        'epoch': epoch,
        'stage': stage_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    filename = os.path.join(out_dir, f"{prefix}_stage{stage_idx}_epoch{epoch}.pth")
    torch.save(checkpoint, filename)
    return filename

def save_final_model(model, optimizer, epoch, stage_idx, out_dir, curriculum, prefix='final'):

    ensure_dir(out_dir)
    final = {
        'epochs': epoch,
        'stages': stage_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    filename = os.path.join(out_dir, f"{prefix}_{curriculum}.pth")
    torch.save(final, filename)
    return filename
