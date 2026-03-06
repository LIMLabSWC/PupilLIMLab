from pathlib import Path
import cv2
import torch
from . detectron2.config import get_cfg
from . detectron2.engine import DefaultPredictor
from . detectron2.utils.visualizer import Visualizer




class Inference:
    def __init__(self, config_path: str, model_path: str, **kwargs):
        """
        Initializes the Inference class.

        Args:
            config_path (str): Path to the configuration file.
            image_path (str): Path to the directory containing the images.
        """
        self.config = None
        self.im_out_dir = kwargs.get('im_out_dir')

        cuda_available = torch.cuda.is_available()

        #Check if MPS is available
        mps_available = torch.backends.mps.is_available()
        if cuda_available:
            self.device = 'cuda'
        elif mps_available:
            self.device = 'mps'
        else:
            #If none set device as CPU
            self.device = 'cpu'
        
        # if self.device != 'cuda':
            # raise NotImplementedError("CUDA is not available")
        # self.device = 'cpu'
        print(f"Using device: {self.device}")
        
        self.predictor = self.get_predictor(config_path, model_path)

    def get_predictor(self, cfg_path: str, model_path) -> DefaultPredictor:
        """
        Returns a DefaultPredictor instance based on the provided configuration file.

        Args:
            cfg_path (str): Path to the configuration file.

        Returns:
            DefaultPredictor: The DefaultPredictor instance.
        """
        # Fetch the config from the given path
        cfg = get_cfg()
        cfg.merge_from_file(cfg_path)
        # Inference should use the config with parameters that are used in training
        # cfg now already contains everything we've set previously. We changed it a little bit for inference:
        cfg.MODEL.WEIGHTS = model_path  # path to the model we just trained
        cfg.MODEL.DEVICE = self.device #configuring the device for inference
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9   # set a custom testing threshold

        return DefaultPredictor(cfg)


    def infer_image_display(self, output, img: str,out_dir=None, out_name=None, ):
        """
        Displays the predictions on a single image.

        Args:
            image_path (str): Path to the image file.
        """


        # Image Path
        # Reading the Image
        if  isinstance(img,(str,Path)):
            raise NotImplementedError("Image path is not supported")
        else:
            image = img
            assert out_name is not None

        # Define custom class names
        class_names = ["Pupil"]

        v = Visualizer(image[:, :, ::-1], metadata = {"thing_classes": class_names}, scale=2.0)
        out = v.draw_instance_predictions(output["instances"].to("cpu"))

        # Convert BGR to RGB
        out_rgb = cv2.cvtColor(out.get_image(), cv2.COLOR_BGR2RGB)

        # Creating a output directory to save predicted images
        #print(f'{output_path= }')
        out_dir =  Path(out_dir)
        if not out_dir.is_dir():
            out_dir.mkdir(parents=True)

        # Saving the image to output directory
        cv2.imwrite(str(out_dir/out_name), out_rgb)
        

    def predict_video(self, video_path, save_images=True,):
        """
        Performs inference on each frame of a video and calculates pupil parameters.

        Args:
            video_path (str): Path to the video file.
            save_images (bool): Whether to save predicted images with bounding boxes.

        Returns:
            list: List of pupil radius values per frame.
        """
        info = {"frame_id": [], "radiusPupil": [], "xCenterPupil": [], "yCenterPupil": []}
        pupil_radius_list = []

        cap = cv2.VideoCapture(video_path,)
        frame_id = 0

        vid_name = Path(video_path).stem

        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            print(f"Processing frame {frame_id}")
            output = self.predictor(frame)

            if frame_id%1000 != 0:
                frame_id += 1

                continue

            if save_images and frame_id%1000 == 0:
                self.infer_image_display(output, frame, self.im_out_dir, f'{vid_name}_{frame_id}.png')

            instances = output["instances"]
            boxes = instances.pred_boxes.tensor.cpu().numpy()

            if len(boxes) > 0:
                classes = instances.pred_classes
                scores = instances.scores

                instances_with_scores = [(i, score) for i, score in enumerate(scores)]
                instances_with_scores.sort(key=lambda x: x[1], reverse=True)

                for index, score in instances_with_scores:
                    if classes[index] == 0:  # 0 is Pupil
                        pupil = boxes[index]
                        pupil_info = get_center_and_radius(pupil)
                        info["frame_id"].append(frame_id)
                        info["radiusPupil"].append(pupil_info["radius"])
                        pupil_radius_list.append(pupil_info["radius"])
                        info["xCenterPupil"].append(int(pupil_info["xCenter"]))
                        info["yCenterPupil"].append(int(pupil_info["yCenter"]))
                        break  # Only one prediction per frame

            frame_id += 1

        cap.release()
        print(f"Processed {frame_id} frames. Found {len(pupil_radius_list)} pupil instances.")
        return pupil_radius_list

def get_center_and_radius(bbox):
    """
    Calculates the center and radius of a bounding box.

    Args:
        bbox (numpy.ndarray): A bounding box represented as [x1, y1, x2, y2].

    Returns:
        dict: A dictionary containing the center (xCenter, yCenter) and radius of the bounding box.
    """
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    xCenter = (bbox[2] + bbox[0]) / 2
    yCenter = (bbox[3] - height / 2)
    radius = width / 2
    height = height / 2

    return {"xCenter": xCenter, "yCenter": yCenter, "radius": radius, "height": height}