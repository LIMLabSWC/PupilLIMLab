import argparse
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
# import cv2
import numpy as np

import pandas as pd
import skvideo.io
import time
import signal  # <-- added for signal handling
import sys     # <-- for sys.exit

# import torch
from tqdm import tqdm
import yaml
from xdetectioncore.paths import posix_from_win # Use the core utility!

from . inference_pupil_sense import Inference, get_center_and_radius


# Global flag for SIGTERM
terminate_flag = False

def sigterm_handler(signum, frame):
    global terminate_flag
    print("SIGTERM received. Will terminate after current frame.")
    terminate_flag = True

# Register SIGTERM handler
signal.signal(signal.SIGTERM, sigterm_handler)

def main(eye_video_paths,invert_gray_im, **kwargs):
    for eye_video_path in eye_video_paths:
        out_dir = Path(eye_video_path).parent/'sample_detection'
        if not out_dir.exists():
            out_dir.mkdir(parents=False, exist_ok=True)
        
        infer = Inference(str(kwargs.get('config_path')),str(kwargs.get('model_path')),
                          im_out_dir=out_dir)
        
        print('loading')
        # time loading video
        load_start = time.time()
        # load video using skvideo.io.vread
        num_frames_to_load = kwargs.get('num_frames',0) 
        
        eye_video = skvideo.io.vread(str(eye_video_path), num_frames=num_frames_to_load, outputdict={'-pix_fmt': 'gray'}) 
        load_end = time.time()
        print(f'load time: {round((load_end-load_start)/60,2)} mins')
        # eye_video = eye_video.dtype(np.uint8)
        # read each frame of video and run pupil detectors
        eye_csvname = (eye_video_path.with_stem(f'{eye_video_path.stem}_eye_ellipse')).with_suffix('.csv')
        # if eye_csvname.exists():
            # print(f'eye_csvname: {eye_csvname} already exists, skipping')
            # continue
        time_pre = time.time()

        ellipse_output = []
        
        global terminate_flag  # use the global flag
        frames_processed = 0  # count processed frames
        
        for frame_number, eye_frame in tqdm(enumerate(eye_video), total=len(eye_video), 
                                            desc='Processing frames'):
            if terminate_flag:
                print("Terminating processing loop due to SIGTERM.")
                break

            # run detector on video frame
            output = infer.predictor(eye_frame)
            instances = output["instances"]
            
            boxes = instances.pred_boxes.tensor.cpu().numpy()

            if len(boxes) <= 0:
                ellipse_output.append([np.nan, np.nan, np.nan,np.nan, np.nan, np.nan])
                frames_processed += 1
                continue


            if frame_number % 10000 == 0:
                infer.infer_image_display(infer.predictor(eye_frame), eye_frame, infer.im_out_dir, f'{frame_number}.png')
            
            classes = instances.pred_classes
            scores = instances.scores

            instances_with_scores = [(i, score) for i, score in enumerate(scores)]
            instances_with_scores.sort(key=lambda x: x[1], reverse=True)

            for pred_i, (index, score) in enumerate(instances_with_scores):
                if classes[index] == 0:  # 0 is Pupil
                    pupil = boxes[index]
                    pupil_info = get_center_and_radius(pupil)
                    radius = int(pupil_info["radius"])
                    height = int(pupil_info["height"])
                    if radius/height > 2:
                        if pred_i == len(instances_with_scores) - 1:
                            # If this is the last instance, append NaN values
                            ellipse_output.append([frame_number, np.nan, np.nan, np.nan, np.nan, np.nan])
                    else:
                        xc = int(pupil_info["xCenter"])
                        yc = int(pupil_info["yCenter"])
                        ellipse_output.append([frame_number, radius, height, xc, yc, float(score)])
                        break  # Only one prediction per frame
            frames_processed += 1

        # Save results even if terminated early
        pupil_est_df = pd.DataFrame(np.array(ellipse_output), columns=['frame_num','radius','height','xc','yc','score'])
        pupil_est_df.to_csv(eye_csvname, index=False)
        print(f'total time taken: {round((time.time()-time_pre)/60,2)} mins')
        
        # Write termination log if terminated
        if terminate_flag:
            log_path = out_dir / f"{eye_video_path.stem}_terminated.txt"
            with open(log_path, "w") as f:
                f.write(f"video_path: {eye_video_path}\n")
                f.write(f"frames_processed: {frames_processed}\n")
            print(f"Termination log saved to {log_path}")
            # Optionally exit after first video if running multiple
            sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("eye_video_paths")
    parser.add_argument("--invert",default=0,type=int)
    parser.add_argument("--pupilsense_config_file",default='configs/pupil_sense.yaml',type=str)

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

    main(eye_video_paths, args.invert,
         config_path=config_path, model_path=model_path, num_frames=num_frames)

