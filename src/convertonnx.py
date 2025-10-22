"""
Convert sript for Video Anomaly Detection & Localization model.

Usage:
    python src/convertonnx.py
"""


import torch
from src.model import ModelDL
from src.config import build_config
from src.dataset import load_data
import onnx
import onnxruntime

cfg = build_config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
val_loader, test_loader = load_data

def convert():
    model = ModelDL(cfg=cfg, device='cuda')
    model = model.to(device=device)
    state_dict = torch.load(cfg.save_path)
    model_state_dict = state_dict['model_state_dict']
    model.load_state_dict(model_state_dict)
    val_iter = iter(val_loader)
    val_item = next(val_iter)
    onnx_program = torch.onnx.export(model,
                                     (val_item[0], val_item[1]),
                                     dynamo = True)
    onnx_program.save(cfg.onnx_model_save_path)
    onnx_model = onnx.load(cfg.onnx_model_save_path)
    print("Checking ONNX Model")
    onnx.checker.check_model(onnx_model)
    print("Done Checking")

if __name__=="__main__":
    convert()
