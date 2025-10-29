import cv2
import time
import queue
import threading
import numpy as np
import torch
import onnxruntime
from torchvision import transforms
from src.config import build_config

config = build_config()

CHUNK_SIZE = config.chunk_size
FRAME_QUEUE_MAX = config.frame_queue_max
RESULT_QUEUE_MAX = config.result_queue_max
HEATMAP_THRESHOLD = config.heatmap_threshold
ANOMALY_PROB_THRESHOLD = config.anomaly_prob_threshold
MERGE_DISTANCE_THRESHOLD = config.merge_distance_threshold

frames_queue = queue.Queue(maxsize=FRAME_QUEUE_MAX)
results_queue = queue.Queue(maxsize=RESULT_QUEUE_MAX)

def video_scores_from_frame_scores(frame_scores, seq_lens):
    B, T = frame_scores.shape
    video_scores = []
    seq_lens_iter = [int(x) for x in seq_lens]
    for i in range(B):
        valid_len = seq_lens_iter[i]
        valid = frame_scores[i, :valid_len]
        k_time = max(1, valid_len // 16)
        topk_vals, _ = torch.topk(torch.tensor(valid), k=k_time)
        video_scores.append(topk_vals.mean())
    return torch.stack(video_scores, dim=0)

def heatmap_to_mask(heatmap_2d, out_w, out_h, threshold=-0.29):
    resized = cv2.resize(heatmap_2d, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
    mask = (resized >= threshold).astype(np.uint8) * 255
    return mask, resized

def merge_contours(contours, distance_threshold=90):
    contours = list(contours)
    if not contours:
        return []
    merged = []
    while contours:
        base = contours.pop(0)
        to_merge = [base]
        i = 0
        while i < len(contours):
            bx, by, bw, bh = cv2.boundingRect(base)
            ox, oy, ow, oh = cv2.boundingRect(contours[i])
            if abs(bx - ox) < distance_threshold and abs(by - oy) < distance_threshold:
                to_merge.append(contours.pop(i))
            else:
                i += 1
        merged.append(np.concatenate(to_merge))
    return merged

def largest_bbox_from_mask(mask_uint8, merge_distance_threshold=90):
    if mask_uint8 is None or mask_uint8.max() == 0:
        return None
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    merged = merge_contours(contours, merge_distance_threshold)
    if not merged:
        return None
    largest = max(merged, key=cv2.contourArea)
    return cv2.boundingRect(largest)

def frame_loader(video_path):
    cap = cv2.VideoCapture(video_path)
    print(f"[Loader] Started reading {video_path}")
    while True:
        frames = []
        for _ in range(CHUNK_SIZE):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        if not frames:
            break
        frames_queue.put(frames)  # blocks if full
    cap.release()
    frames_queue.put(None)
    print("[Loader] Finished loading video")

def inference_worker(sess, preprocess):
    CHUNK_SIZE = 256
    chunk_idx = 0

    while True:
        item = frames_queue.get()
        if item is None:
            print("[Inference] Received end signal.")
            break

        frames = item
        pre_start = time.time()

        # --- Zero-pad last chunk if shorter ---
        if len(frames) < CHUNK_SIZE:
            pad_frames = CHUNK_SIZE - len(frames)
            pad_frame = np.zeros_like(frames[0])
            for _ in range(pad_frames):
                frames.append(pad_frame.copy())
            print(f"[Inference] Zero-padded last chunk with {pad_frames} frames (now {len(frames)} total).")

        # --- Preprocess ---
        processed = []
        for f in frames:
            f_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            tensor = preprocess(f_rgb).numpy()
            processed.append(tensor)
        frames_np = np.stack(processed)
        frames_np = np.expand_dims(frames_np, axis=0)  # Add batch dim
        seq_len = np.array([CHUNK_SIZE], dtype=np.int64)

        feed = {}
        for name in [i.name for i in sess.get_inputs()]:
            if "x" in name.lower():
                feed[name] = frames_np
            elif "seq" in name.lower():
                feed[name] = seq_len

        pre_end = time.time()
        infer_start = time.time()
        outputs = sess.run(None, feed)
        infer_end = time.time()

        # --- Extract outputs ---
        frame_logit, heatmap_logit = outputs  # [B, T, 1] and [B, T, 1, H, W]

        # --- Safely reshape for scoring ---
        frame_tensor = torch.from_numpy(frame_logit)
        if frame_tensor.ndim == 3 and frame_tensor.shape[-1] == 1:
            frame_tensor = frame_tensor.squeeze(-1)  # → [B, T]
        elif frame_tensor.ndim == 2:
            pass  # already [B, T]
        else:
            raise ValueError(f"Unexpected frame_logit shape: {frame_tensor.shape}")

        seq_tensor = torch.tensor(seq_len)
        video_score = video_scores_from_frame_scores(frame_tensor, seq_tensor)

        print(f"[Inference] Chunk {chunk_idx:03d} video_score: {video_score.item():.4f}")

        # Pack timing info
        infer_time = infer_end - infer_start
        pre_time = pre_end - pre_start
        post_time = 0.0  # Will be updated later if needed

        timing_info = {"pre_time": pre_time, "post_time": post_time}

        # These 2 are not timing info, but I included the inference result in here to make it easier for me
        timing_info["video_score"] = float(video_score.item())
        timing_info["heatmap_logit"] = heatmap_logit

        results_queue.put((frames, timing_info, infer_time, chunk_idx))
        chunk_idx += 1
        frames_queue.task_done()

    results_queue.put(None)
    print("[Inference] Finished all chunks.")


def display_worker():
    CHUNK_SIZE = 256
    FRAME_DELAY_MS = 33  # ~30 FPS playback target

    while True:
        item = results_queue.get()
        if item is None:
            print("[Display] Received end signal.")
            break

        # Unpack
        frames, timing_info, infer_time, chunk_idx = item
        chunk_len = len(frames)

        # --- Extract timing info ---
        pre_time = timing_info.get("pre_time", 0.0)
        post_time = timing_info.get("post_time", 0.0)
        display_time = chunk_len * FRAME_DELAY_MS / 1000
        total_chunk_time = pre_time + infer_time + post_time + display_time
        score = timing_info.get("video_score", 0.0)
        is_anomaly = (score >= 0.0)

        print(
            f"[Chunk Summary {chunk_idx:03d}] "
            f"Score: {score:.4f} "
            f"Pre: {pre_time:.2f}s | Infer: {infer_time:.2f}s | "
            f"Post+Draw: {post_time:.2f}s | Display: {display_time:.2f}s "
            f"| Total: {total_chunk_time:.2f}s"   
        )

        heatmap = timing_info.get("heatmap_logit", None)

        for t, frame in enumerate(frames):
            # display_frame = frame.copy()

            # --- Heatmap Overlay (if provided) ---
            if heatmap is not None and t < heatmap.shape[1]:
                hm = heatmap[0, t, 0]  # shape (H, W)

                mask_uint8, hm_resized = heatmap_to_mask(
                    hm,
                    out_w = frames[t].shape[1],
                    out_h = frames[t].shape[0],
                    threshold = HEATMAP_THRESHOLD
                )

                if is_anomaly:
                    bbox = largest_bbox_from_mask(mask_uint8, merge_distance_threshold = MERGE_DISTANCE_THRESHOLD)
                    if bbox is not None:
                        x, y, w, h = bbox
                        cv2.rectangle(frames[t], (x, y), (x + w, y + h), (0, 255, 0), 2)

            # --- Display frame ---
            cv2.imshow("Live Feed + Heatmap", frame)
            if cv2.waitKey(FRAME_DELAY_MS) & 0xFF == ord('q'):
                print("[Display] Quit signal received.")
                results_queue.task_done()
                cv2.destroyAllWindows()
                return

        results_queue.task_done()

    cv2.destroyAllWindows()
    print("[Display] Closed all windows.")


# ---------------- Main ----------------
def run_async_visualization(model_path, video_path):
    sess_options = onnxruntime.SessionOptions()
    sess_options.enable_mem_pattern = True
    sess_options.enable_mem_reuse = True
    sess = onnxruntime.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_names = [i.name for i in sess.get_inputs()]

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    loader_t = threading.Thread(target=frame_loader, args=(video_path,))
    infer_t = threading.Thread(target=inference_worker, args=(sess,preprocess))
    display_t = threading.Thread(target=display_worker)

    loader_t.start()
    infer_t.start()
    display_t.start()

    loader_t.join()
    infer_t.join()
    display_t.join()
    time.sleep(0.5)
    cv2.destroyAllWindows()
    print("[Main] All threads completed.")

if __name__ == "__main__":
    run_async_visualization(
        model_path=config.onnx_model_save_path,
        video_path="../../../CLIP/UCFCrimes/Videos/Explosion/Explosion051_x264.mp4" #You can use any video files
    )
