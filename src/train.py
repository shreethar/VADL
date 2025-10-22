"""
Train sript for Video Anomaly Detection & Localization model.

Usage:
    python src/train.py
"""


import wandb
import torch
from src.model import ModelDL
from src.config import build_config
from src.helper import combined_criterion, video_scores_from_frame_scores
from src.dataset import load_data
import time
import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from torch.nn.utils import clip_grad_norm_

cfg = build_config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ModelDL(cfg=cfg, device='cuda')
model = model.to(device=device)

train_loader, val_loader = load_data()

optimizer = torch.optim.AdamW(
    [{"params": model.classifier.parameters(), "lr": cfg.lr_detect},  # detection head
    {"params": model.pre_temporal_spatial_head.parameters(), "lr": cfg.lr_localize},
    {"params": model.post_temporal_spatial_head.parameters(), "lr": cfg.lr_localize}] # localization head]
)

config = {
    "learning_rate_detection": cfg.lr_detect,
    "learning_rate_localization": cfg.lr_localize,
    "architecture": "your-model-name",
    "dataset": cfg.dataset,
    "epochs": cfg.epochs,
    "patience": cfg.patience,
    "accum_steps": cfg.accum_steps,
    "lambda_class_weight": 0.75,
    "lambda_heatmap_weight": 5.0,
    "lambda_tv": 0.2,
    "lambda_temp": 0.75,
    "lambda_area": 0.01,
    "lambda_consistency": 1.5,
    "lambda_frame_consistency": 0.1,
    "dropout_pre": 0.4,
    "dropout_post": 0.0
}

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-6)

# --- training state ---
best_loss = np.inf
no_improve_epochs = 0

def train_one_epoch(model, dataloader, optimizer, criterion, cfg, device):
    model.train()
    metrics = {}
    return metrics

def validate_one_epoch(model, dataloader, criterion, cfg, device):
    model.eval()
    metrics = {}
    return metrics


def train():
    with wandb.init(project=cfg.project_name, config=config) as run:

        best_loss = float("inf")
        no_improve_epochs = 0

        for epoch in range(config['epochs']):
            epoch_start = time.time()
            model.train()

            # ---------- running sums for TRAIN ----------
            running_train_loss = 0.0
            running_train_detection_loss = 0.0
            running_train_heatmap_loss = 0.0
            running_train_tv_loss = 0.0
            running_train_temp_loss = 0.0
            running_train_area_loss = 0.0
            running_train_consistency_mse_loss = 0.0
            running_train_consistency_bce_loss = 0.0
            running_train_consistency_loss = 0.0
            running_train_video_score_hm_mean = 0.0
            running_train_video_score_cls_mean = 0.0

            train_video_label_ground_truth = []
            train_video_label_prediction = []

            optimizer.zero_grad()

            train_iter = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Epoch {epoch+1}/{config['epochs']} Train",
                leave=False
            )

            for step, (video_frames, video_length, video_label, video_path) in train_iter:
                video_frames = video_frames.to(device)
                video_length = video_length.to(device)
                video_label = video_label.to(device).long()

                binary_logits, heatmaps = model(video_frames, video_length)
                raw_loss, loss_dict = combined_criterion(video_label, binary_logits, heatmaps)
                scaled_loss = raw_loss / cfg.accum_steps
                scaled_loss.backward()

                with torch.no_grad():
                    video_score_logits = video_scores_from_frame_scores(
                        binary_logits[:, :, 0].detach(), video_length, video_label
                    )
                    video_pred_binary = (torch.sigmoid(video_score_logits) >= 0.5).int().cpu().tolist()
                    train_video_label_prediction.extend(video_pred_binary)
                    train_video_label_ground_truth.extend(video_label.int().cpu().tolist())

                if (step + 1) % cfg.accum_steps == 0:
                    clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=5.0)
                    optimizer.step()
                    optimizer.zero_grad()

                # accumulate per loss
                running_train_loss += float(raw_loss.item())
                running_train_detection_loss += float(loss_dict['loss_classifier'])
                running_train_heatmap_loss += float(loss_dict['loss_heatmap'])
                running_train_tv_loss += float(loss_dict['tv_loss'])
                running_train_temp_loss += float(loss_dict['temp_loss'])
                running_train_area_loss += float(loss_dict['area_loss'])
                running_train_consistency_mse_loss += float(loss_dict['consistency_mse'])
                running_train_consistency_bce_loss += float(loss_dict['consistency_frame_bce'])
                running_train_consistency_loss += float(loss_dict['consistency_total'])
                running_train_video_score_hm_mean += float(loss_dict['video_score_hm_mean'])
                running_train_video_score_cls_mean += float(loss_dict['video_score_cls_mean'])

            if (step + 1) % cfg.accum_steps != 0:
                clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad()

            n_train_batches = len(train_loader)
            train_loss = running_train_loss / n_train_batches
            train_det_loss = running_train_detection_loss / n_train_batches
            train_hm_loss = running_train_heatmap_loss / n_train_batches
            train_tv_loss = running_train_tv_loss / n_train_batches
            train_temp_loss = running_train_temp_loss / n_train_batches
            train_area_loss = running_train_area_loss / n_train_batches
            train_cons_mse = running_train_consistency_mse_loss / n_train_batches
            train_cons_bce = running_train_consistency_bce_loss / n_train_batches
            train_cons_total = running_train_consistency_loss / n_train_batches
            train_vscore_hm = running_train_video_score_hm_mean / n_train_batches
            train_vscore_cls = running_train_video_score_cls_mean / n_train_batches

            train_f1_score = f1_score(train_video_label_ground_truth, train_video_label_prediction)
            train_accuracy = accuracy_score(train_video_label_ground_truth, train_video_label_prediction)

            # ---------- VALIDATION ----------
            model.eval()
            running_val_loss = 0.0
            running_val_detection_loss = 0.0
            running_val_heatmap_loss = 0.0
            running_val_tv_loss = 0.0
            running_val_temp_loss = 0.0
            running_val_area_loss = 0.0
            running_val_consistency_mse_loss = 0.0
            running_val_consistency_bce_loss = 0.0
            running_val_consistency_loss = 0.0
            running_val_video_score_hm_mean = 0.0
            running_val_video_score_cls_mean = 0.0

            validation_video_label_ground_truth = []
            validation_video_label_prediction = []

            with torch.no_grad():
                val_iter = tqdm(
                    enumerate(val_loader),
                    total=len(val_loader),
                    desc=f"Epoch {epoch+1}/{config['epochs']} Val",
                    leave=False
                )

                for vstep, (video_frames, video_length, video_label, video_path) in val_iter:
                    video_frames = video_frames.to(device)
                    video_length = video_length.to(device)
                    video_label = video_label.to(device).long()

                    binary_logits, heatmaps = model(video_frames, video_length)
                    v_loss, v_dict = combined_criterion(video_label, binary_logits, heatmaps)

                    running_val_loss += float(v_loss.item())
                    running_val_detection_loss += float(v_dict['loss_classifier'])
                    running_val_heatmap_loss += float(v_dict['loss_heatmap'])
                    running_val_tv_loss += float(v_dict['tv_loss'])
                    running_val_temp_loss += float(v_dict['temp_loss'])
                    running_val_area_loss += float(v_dict['area_loss'])
                    running_val_consistency_mse_loss += float(v_dict['consistency_mse'])
                    running_val_consistency_bce_loss += float(v_dict['consistency_frame_bce'])
                    running_val_consistency_loss += float(v_dict['consistency_total'])
                    running_val_video_score_hm_mean += float(v_dict['video_score_hm_mean'])
                    running_val_video_score_cls_mean += float(v_dict['video_score_cls_mean'])

                    video_score_logits = video_scores_from_frame_scores(
                        binary_logits[:, :, 0].detach(), video_length, video_label
                    )
                    video_pred_binary = (torch.sigmoid(video_score_logits) >= 0.5).int().cpu().tolist()
                    validation_video_label_prediction.extend(video_pred_binary)
                    validation_video_label_ground_truth.extend(video_label.int().cpu().tolist())

            n_val_batches = len(val_loader)
            val_loss = running_val_loss / n_val_batches
            val_det_loss = running_val_detection_loss / n_val_batches
            val_hm_loss = running_val_heatmap_loss / n_val_batches
            val_tv_loss = running_val_tv_loss / n_val_batches
            val_temp_loss = running_val_temp_loss / n_val_batches
            val_area_loss = running_val_area_loss / n_val_batches
            val_cons_mse = running_val_consistency_mse_loss / n_val_batches
            val_cons_bce = running_val_consistency_bce_loss / n_val_batches
            val_cons_total = running_val_consistency_loss / n_val_batches
            val_vscore_hm = running_val_video_score_hm_mean / n_val_batches
            val_vscore_cls = running_val_video_score_cls_mean / n_val_batches

            val_f1_score = f1_score(validation_video_label_ground_truth, validation_video_label_prediction)
            val_accuracy = accuracy_score(validation_video_label_ground_truth, validation_video_label_prediction)

            # ---------- W&B log ----------
            run.log({
                "epoch": epoch + 1,
                "lr": optimizer.param_groups[0]['lr'],
                "time/epoch_sec": time.time() - epoch_start,

                "train/loss": train_loss,
                "train/detection_loss": train_det_loss,
                "train/heatmap_loss": train_hm_loss,
                "train/tv_loss": train_tv_loss,
                "train/temp_loss": train_temp_loss,
                "train/area_loss": train_area_loss,
                "train/consistency_mse": train_cons_mse,
                "train/consistency_bce": train_cons_bce,
                "train/consistency_total": train_cons_total,
                "train/video_score_hm_mean": train_vscore_hm,
                "train/video_score_cls_mean": train_vscore_cls,
                "train/f1": train_f1_score,
                "train/accuracy": train_accuracy,

                "val/loss": val_loss,
                "val/detection_loss": val_det_loss,
                "val/heatmap_loss": val_hm_loss,
                "val/tv_loss": val_tv_loss,
                "val/temp_loss": val_temp_loss,
                "val/area_loss": val_area_loss,
                "val/consistency_mse": val_cons_mse,
                "val/consistency_bce": val_cons_bce,
                "val/consistency_total": val_cons_total,
                "val/video_score_hm_mean": val_vscore_hm,
                "val/video_score_cls_mean": val_vscore_cls,
                "val/f1": val_f1_score,
                "val/accuracy": val_accuracy,
            })

            # ---------- checkpointing ----------
            if not np.isnan(val_loss):
                if val_loss < best_loss + 1e-6:
                    best_loss = val_loss
                    no_improve_epochs = 0
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduuler_state_dict': scheduler.state_dict(),
                        'val_loss': val_loss
                    }, cfg.save_path)
                    print(f"Saved new best model (val_loss={val_loss:.4f}) to {cfg.save_path}")
                else:
                    no_improve_epochs += 1
                    print(f"No improvement for {no_improve_epochs} epochs (best_loss={best_loss:.4f})")
                epoch_duration = time.time() - epoch_start
                print(f"Epoch {epoch+1} completed in {epoch_duration:.2f}s | val_loss={val_loss:.4f}")

            if no_improve_epochs >= cfg.patience:
                print(f"Early stopping triggered after {no_improve_epochs} epochs without improvement.")
                break

        print("Training finished. Best loss:", best_loss)

if __name__=="__main__":
    train()