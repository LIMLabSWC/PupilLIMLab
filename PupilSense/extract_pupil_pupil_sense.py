import os

import platform

from cv2.gapi import infer
import psutil

from PupilSense.ellipse_fitting import run_ransac_gpu_pytorch, fit_ellipse_direct_gpu
if platform.system() == 'Windows':
    # Add FFmpeg binaries
    os.add_dll_directory(r"C:\Program Files\ffmpeg\bin")

    # Add CUDA binaries 
    os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\bin")


import argparse
from pathlib import Path
import numpy as np

import pandas as pd

import signal  # <-- added for signal handling

# import torch
import torch.nn.functional as F
from tqdm import tqdm
import yaml
from xdetectioncore.paths import posix_from_win # Use the core utility!

from inference_pupil_sense import Inference, get_center_and_radius

import torch
import decord
import cv2
decord.bridge.set_bridge('torch')

# Global flag for SIGTERM
terminate_flag = False

def sigterm_handler(signum, frame):
    global terminate_flag
    print("SIGTERM received. Will terminate after current frame.")
    terminate_flag = True


def kill_zombie_python_processes():
    """Kills other python processes to clear GPU memory (Cross-Platform)."""
    current_pid = os.getpid()
    count = 0
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            # Check if it's a python process and NOT this current script
            if 'python' in proc.info['name'].lower() and proc.info['pid'] != current_pid:
                proc.kill()
                count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if count > 0:
        print(f"Pre-launch cleanup: {count} zombie processes terminated.")
        

# Register SIGTERM handler
signal.signal(signal.SIGTERM, sigterm_handler)

def process_instance(instance, frame_idx, info_list):
    """
    Fits ellipse and appends results to info_list in order:
    ['frame_num', 'radius', 'height', 'xc', 'yc', 'score', 'angle']
    """
    scores = instance.scores
    best_idx = torch.argmax(scores) if torch.is_tensor(scores) else np.argmax(scores)
    
    mask = instance.pred_masks[best_idx]
    bbox = instance.pred_boxes.tensor[best_idx]
    score = float(scores[best_idx])
    
    # Handle coordinate types (GPU Long vs CPU Int)
    x1, y1, x2, y2 = (bbox.long() if torch.is_tensor(bbox) else bbox.astype(int))
    
    # Fit Ellipse on ROI
    roi_mask = mask[y1:y2, x1:x2]
    params = None
    if roi_mask.any():
        # Ensure ROI is on GPU for the mathematical fitter
        gpu_roi = roi_mask if torch.is_tensor(roi_mask) else torch.from_numpy(roi_mask).cuda()
        params = fit_ellipse_direct_gpu(gpu_roi)

    if params is not None and not torch.isnan(params[0]):
        p = params.cpu().numpy() if torch.is_tensor(params) else params
        # Columns: ['frame_num','radius','height','xc','yc','score','angle']
        info_list.append([
            int(frame_idx), float(p[2]), float(p[3]), 
            float(p[0] + x1), float(p[1] + y1), 
            score, float(p[4])
        ])
    else:
        info_list.append([int(frame_idx), np.nan, np.nan, np.nan, np.nan, score, np.nan])
            
    return params, x1, y1


def process_instance_gpu(instance, frame_idx, info_list):
    # This stays entirely on GPU
    mask = instance.pred_masks[0] # Best detection
    bbox = instance.pred_boxes.tensor[0]
    
    # Crop to pupil area
    x1, y1, x2, y2 = bbox.long()
    roi_mask = mask[y1:y2, x1:x2]
    
    params = fit_ellipse_direct_gpu(roi_mask)
    
    if params is not None:
        xc, yc, a, b, angle = params
        # Log results (move only these 5 numbers back to CPU)
        info_list.append([
            frame_idx, a.item(), b.item(), 
            (xc + x1).item(), (yc + y1).item(), 
            instance.scores[0].item(), angle.item()
        ])
    return params, x1, y1


def save_snapshot(frame_idx, frame_bgr, output, params, x1, y1, infer_obj):
    # Move to CPU only for the periodic image save
    snap_img = frame_bgr.cpu().numpy().copy() if torch.is_tensor(frame_bgr) else frame_bgr.copy()
    if params is not None:
        p = params.cpu().numpy() if torch.is_tensor(params) else params
        if not np.isnan(p[0]):
            center = (int(p[0] + x1), int(p[1] + y1))
            axes = (int(p[2]), int(p[3]))
            cv2.ellipse(snap_img, center, axes, float(p[4]), 0, 360, (0, 255, 0), 1)
    infer_obj.infer_image_display(output, snap_img, infer_obj.im_out_dir, f'{frame_idx}.png')

def main(eye_video_paths, invert_gray_im, **kwargs):
    for eye_video_path in eye_video_paths:
        out_dir = Path(eye_video_path).parent / 'sample_detection'
        out_dir.mkdir(parents=True, exist_ok=True)
        infer = Inference(str(kwargs['config_path']), str(kwargs['model_path']), im_out_dir=out_dir)
        
        vr = decord.VideoReader(str(eye_video_path), ctx=decord.cpu(0))
        ellipse_output = []

        for i in tqdm(range(len(vr)), desc="Sequential CPU Loading"):
            if terminate_flag: break
            
            # Load frame to CPU numpy array
            frame_cpu = vr[i].numpy() 
            eye_frame_gray = cv2.cvtColor(frame_cpu, cv2.COLOR_RGB2GRAY)
            eye_frame_bgr = cv2.cvtColor(eye_frame_gray, cv2.COLOR_GRAY2BGR)

            # Inference
            output = infer.predictor(eye_frame_bgr)
            instances = output["instances"]
            params, x1, y1 = None, 0, 0

            if len(instances) > 0:
                params, x1, y1 = process_instance(instances, i, ellipse_output)
            else:
                ellipse_output.append([i, np.nan, np.nan, np.nan, np.nan, 0.0, np.nan])

            if i % 10000 == 0:
                save_snapshot(i, eye_frame_bgr, output, params, x1, y1, infer)

        df = pd.DataFrame(ellipse_output, columns=['frame_num','radius','height','xc','yc','score','angle'])
        df.to_csv(out_dir / f"{Path(eye_video_path).stem}_pupil.csv", index=False)


def main_gpu(eye_video_paths, invert_gray_im, **kwargs):
    for eye_video_path in eye_video_paths:
        out_dir = Path(eye_video_path).parent / 'sample_detection'
        out_dir.mkdir(parents=True, exist_ok=True)
        
        infer = Inference(str(kwargs['config_path']), str(kwargs['model_path']), im_out_dir=out_dir)
        # Use GPU context for decord
        vr = decord.VideoReader(str(eye_video_path), ctx=decord.gpu(0))
        
        ellipse_output = []
        batch_size = 8 
        weights = torch.tensor([0.299, 0.587, 0.114], device='cuda', dtype=torch.half).view(1, 1, 1, 3)

        for start_idx in tqdm(range(0, len(vr), batch_size), desc="Batch GPU Processing"):
            if terminate_flag: break
            
            indices = list(range(start_idx, min(start_idx + batch_size, len(vr))))
            batch_frames = vr.get_batch(indices) # [B, H, W, 3] on GPU
            B, H, W, _ = batch_frames.shape

            with torch.no_grad():
                # 1. Convert to Half Precision and Grayscale immediately
                # Using .half() saves massive VRAM during the upscale
                
                # 2. Process in Channels Last format for tensor core optimization
                gray = (batch_frames.half() * weights).sum(dim=-1, keepdim=True)
                batch_input = gray.expand(-1, -1, -1, 3).permute(0, 3, 1, 2) # [B, 3, H, W]
                
                # 3. Optimized Upscale
                scale = 800.0 / min(H, W)
                new_h, new_w = int(H * scale), int(W * scale)
                
                batch_input = torch.nn.functional.interpolate(
                    batch_input, 
                    size=(new_h, new_w), 
                    mode='bilinear', 
                    align_corners=False
                )
            outputs = infer.predict_batch([{"image": img, "height": H, "width": W} for img in batch_input])
            
            for i, output in enumerate(outputs):
                frame_idx = start_idx + i
                instances = output["instances"]
                params, x1, y1 = None, 0, 0
                if len(instances) > 0:
                    params, x1, y1 = process_instance_gpu(instances, frame_idx, ellipse_output)
                else:
                    ellipse_output.append([frame_idx, np.nan, np.nan, np.nan, np.nan, 0.0, np.nan])

                if frame_idx % 10000 == 0:
                    save_snapshot(frame_idx, batch_frames[i], output, params, x1, y1, infer)

        df = pd.DataFrame(ellipse_output, columns=['frame_num','radius','height','xc','yc','score','angle'])
        df.to_csv(out_dir / f"{Path(eye_video_path).stem}_pupil.csv", index=False)


if __name__ == "__main__":
    kill_zombie_python_processes()

    parser = argparse.ArgumentParser()
    parser.add_argument("eye_video_paths")
    parser.add_argument("--invert",default=0,type=int)
    parser.add_argument("--pupilsense_config_file",default='configs/pupil_sense.yaml',type=str)
    parser.add_argument("--use_gpu", action='store_true', help='Use GPU for inference if available.')

    print('Running pupil detection on eye video')
    print('----------------------------------')

    # model configs
    args = parser.parse_args()
    with open(args.pupilsense_config_file) as f:
        config = yaml.safe_load(f)

    ceph_dir = Path(config[f'ceph_dir_{platform.system().lower()}'])
    config_path = ceph_dir / posix_from_win(config['config_path'])
    model_path = ceph_dir / posix_from_win(config['model_path'])

    print(f'config_path: {config_path}')
    print(f'model_path: {model_path}')
    num_frames = config['num_frames']

    eye_video_paths = [ceph_dir/ posix_from_win(eye_video_path) 
                       for eye_video_path in args.eye_video_paths.split(';')]

    if args.use_gpu and torch.cuda.is_available():
        print("GPU detected. Running inference on GPU.")
        main_gpu(eye_video_paths, args.invert, 
                 config_path=config_path, model_path=model_path, num_frames=num_frames)
    else:
        print("No GPU detected or --use_gpu not set. Running inference on CPU.")
        main(eye_video_paths, args.invert,
             config_path=config_path, model_path=model_path, num_frames=num_frames)

