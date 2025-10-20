import os
import csv
import json
import torch

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

class MetricsLogger:   # Logs training metrics and saves them to a CSV file.

    def __init__(self, logfile, curriculum_name=None): 
        self.logfile = logfile
        self.curriculum_name = curriculum_name
        self.fields = None
        self.rows = []
        self.dataset_info = None
        self.stages = []

    
    def save_stage_config(self, stage_config, dataset_info=None):
        """Save stage configuration and dataset info (first stage only)"""
        # Store dataset info only once (from first stage)
        if self.dataset_info is None and dataset_info:
            self.dataset_info = dataset_info
        
        # Create simplified stage config with resolved values
        stage_data = dict(stage_config)
        
        # Ensure lr is stored as numeric value, not variable reference
        if 'lr' in stage_data and hasattr(stage_data['lr'], '__call__'):
            # If lr is a function/callable, we can't resolve it here
            stage_data['lr'] = str(stage_data['lr'])
        
        self.stages.append(stage_data)

    def save_metadata(self):
        """Save simplified metadata to JSON file"""
        if not self.stages:
            return
            
        metadata_file = self.logfile.replace('.csv', '_config.json')
        metadata = {
            'curriculum': self.curriculum_name,
            'stages': self.stages
        }
        
        # Add dataset info if available
        if self.dataset_info:
            metadata['dataset'] = self.dataset_info
        
        ensure_dir(os.path.dirname(metadata_file))
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def log(self, data: dict): 
        # Log only training data, dataset context saved separately in JSON
        if self.fields is None:
            self.fields = list(data.keys())
        self.rows.append(data)

    def save(self):
        if not self.rows:
            return
        ensure_dir(os.path.dirname(self.logfile))
        
        # Only include core metrics columns
        ordered_fields = ['stage', 'epoch', 'loss', 'time']
        
        with open(self.logfile, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=ordered_fields)
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
