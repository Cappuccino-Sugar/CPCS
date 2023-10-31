import os
import torch
import shutil

def save_checkpoint(args, state, is_best, filename):
    out = os.path.join(args.save_path, filename)
    torch.save(state, out)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')