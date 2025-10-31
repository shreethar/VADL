"""
Evaluate sript for Video Anomaly Detection & Localization model.

Usage:
    python src/evaluate.py
"""

import torch
from src.model import ModelDL
from src.config import build_config
from src.helper import frame_scores_from_heatmap, video_scores_from_frame_scores
from src.dataset import load_data
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

cfg = build_config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_model():
    model = ModelDL(cfg=cfg, device='cuda')
    model = model.to(device=device)
    
    state_dict = torch.load(cfg.save_path)
    model_state_dict = state_dict['model_state_dict']
    model.load_state_dict(model_state_dict)
    model.eval()
    
    val_loader, test_loader = load_data(training=False)

    val_video_label_prediction = []
    val_video_label_ground_truth = []

    val_iter = tqdm(enumerate(val_loader),
                    total=len(val_loader),
                    leave=False)
    
    with torch.no_grad():
        for step, (video_frames, video_length, video_label, video_path) in val_iter:
            video_frames = video_frames.to(device)
            video_length = video_length.to(device)
            video_label = video_label.to(device).long()
            
            binary_logits, heatmaps = model(video_frames, video_length)
            video_score_logits = video_scores_from_frame_scores(
                binary_logits[:, :, 0].detach(), video_length, video_label
            )
            video_pred_binary = (torch.sigmoid(video_score_logits) >= 0.5).int().cpu().tolist()
            val_video_label_prediction.extend(video_pred_binary)
            val_video_label_ground_truth.extend(video_label.int().cpu().tolist())

    cfm_val = confusion_matrix(val_video_label_ground_truth, val_video_label_prediction)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cfm_val, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(val_video_label_ground_truth), yticklabels=np.unique(val_video_label_ground_truth))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Validation Set Confusion Matrix')
    plt.savefig(cfg.val_cfm_path)
    plt.show()

    test_iter = tqdm(enumerate(test_loader),
                     total=len(test_loader),
                     leave=False)
    
    test_video_label_prediction = []
    test_video_label_ground_truth = []

    with torch.no_grad():
        for step, (video_frames, video_length, video_label, video_path) in test_iter:
            video_frames = video_frames.to(device)
            video_length = video_length.to(device)
            video_label = video_label.to(device).long()
            
            binary_logits, heatmaps = model(video_frames, video_length)
            video_score_logits = video_scores_from_frame_scores(
                binary_logits[:, :, 0].detach(), video_length, video_label
            )
            video_pred_binary = (torch.sigmoid(video_score_logits) >= 0.5).int().cpu().tolist()
            test_video_label_prediction.extend(video_pred_binary)
            test_video_label_ground_truth.extend(video_label.int().cpu().tolist())

    cfm_test = confusion_matrix(test_video_label_ground_truth, test_video_label_prediction)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cfm_test, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(test_video_label_ground_truth), yticklabels=np.unique(test_video_label_ground_truth))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Test Set Confusion Matrix')
    plt.savefig(cfg.test_cfm_path)
    plt.show()


if __name__ == "__main__":
    evaluate_model()
