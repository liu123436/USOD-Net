# 支持源代码已有的检查点格式与返回值。
import torch

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        # Full checkpoint with model, optimizer, and training state
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint.get('epoch', 1)  # Default to epoch 1 if not found
            loss = checkpoint.get('loss', 0)    # Default to 0 if not found
            return epoch, loss
        else:
            # Might be a state_dict directly in the dict
            model.load_state_dict(checkpoint)
            return 1, 0
    else:
        # Assume it's just the model state_dict
        model.load_state_dict(checkpoint)
        return 1, 0
