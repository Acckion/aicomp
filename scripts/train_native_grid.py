"""Audited native-feature worker; ordinary D-FINE losses and EMA."""
import native_grid
from train_ir_content import audited_step
import torch
import train_mechanisms

if __name__ == '__main__':
    torch.optim.AdamW.step = audited_step
    train_mechanisms.main()
