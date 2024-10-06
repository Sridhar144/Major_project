# import argparse
# import time
# import cv2
# import torch
# import torch.backends.cudnn as cudnn
# from datetime import datetime
# from pathlib import Path
# from numpy import random
# from torchvision import transforms
# from models.experimental import attempt_load
# from utils.datasets import LoadStreams, LoadImages, letterbox
# from utils.general import (
#     check_img_size, check_imshow, non_max_suppression, 
#     apply_classifier, scale_coords, xyxy2xywh, 
#     strip_optimizer, set_logging, increment_path, non_max_suppression_kpt
# )
# from utils.plots import plot_one_box, output_to_keypoint, plot_skeleton_kpts
# from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
# import numpy as np

# # Display the video feed in a separate window
# def display_window(img):
#     cv2.imshow("Detection", img)
#     return cv2.waitKey(1) != ord('q')  # press 'q' to quit

# # Load pose model
# def load_pose_model(weights_path, device):
#     model = torch.load(weights_path, map_location=device)['model'].float().eval()
#     return model.to(device)

# # Pose estimation
# def run_pose_estimation(tensor_image, model):
#     tensor_image = tensor_image.half() if next(model.parameters()).dtype == torch.float16 else tensor_image.float()
#     pred = model(tensor_image)
#     return pred[0] if isinstance(pred, tuple) else pred

# # Load violence detection model
# def load_vio_model(weights_path, device):
#     model = attempt_load(weights_path, map_location=device)
#     return model.to(device).eval()

# # Run violence detection
# def run_vio_detection(img, model, device, half=False):
#     img = img.to(device).half() if half else img.float()  
#     img /= 255.0  
#     if img.ndimension() == 3:
#         img = img.unsqueeze(0)

#     with torch.no_grad():  
#         with torch.amp.autocast('cuda', enabled=half):
#             pred = model(img)[0]

#     return non_max_suppression(pred, 0.6, opt.iou_thres)

# def detect(save_img=False):
#     source, weights, view_img, save_txt, imgsz, trace = (
#         opt.source, opt.weights, opt.view_img, 
#         opt.save_txt, opt.img_size, not opt.no_trace
#     )
#     save_img = not opt.nosave and not source.endswith('.txt')
#     webcam = source.isnumeric() or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))

#     save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  
#     (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

#     set_logging()
#     device = select_device(opt.device)
#     half = device.type != 'cpu'

#     # Load pose and violence detection models
#     pose_model = load_pose_model('yolov7-w6-pose.pt', device)
#     model = attempt_load(weights, map_location=device)  
#     stride = int(model.stride.max())
#     imgsz = check_img_size(imgsz, s=stride)  
#     vio_model = load_vio_model('yolov7_vio_detect_v3.pt', device)

#     if trace:
#         model = TracedModel(model, device, opt.img_size)

#     if half:
#         model.half()

#     vid_path, vid_writer = None, None
#     dataset = LoadStreams(source, img_size=imgsz, stride=stride) if webcam else LoadImages(source, img_size=imgsz, stride=stride)

#     names = model.module.names if hasattr(model, 'module') else model.names
#     colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

#     if device.type != 'cpu':
#         model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
        
#     t0 = time.time()

#     for path, img, im0s, vid_cap in dataset:
#         # Run Violence Detection
#         print("Processing Violence Detection...")
#         vio_img = torch.from_numpy(img).to(device)
#         vio_pred = run_vio_detection(vio_img, vio_model, device, half=half)

#         # Handle violence detection results
#         level_sys, violence_count, vio_results_str = handle_vio_detection(vio_pred, vio_img, names, im0s)

#         # Resize image for pose estimation & Run pose estimation
#         if not webcam:
#             print("Processing Pose Estimation...")
#             resized_im0s = letterbox(im0s, 960, stride=64, auto=True)[0]
#             resized_tensor = transforms.ToTensor()(resized_im0s).unsqueeze(0).to(device)
#             pose_pred = run_pose_estimation(resized_tensor, pose_model)
#             pose_pred = output_to_keypoint(pose_pred)
#         else:
#             resized_im0s = im0s

#         # Object detection
#         img = prepare_image(img, device, half)
#         pred = model(img, augment=opt.augment)[0]
#         pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

#         # Handle detections and results
#         obj_results_str = handle_object_detection(pred, resized_im0s, pose_pred, names, im0s)

#         # Print detected time and results
#         print_detection_summary(vio_results_str, obj_results_str)

#         if not display_window(im0s):
#             break

#         # Save images or video
#         save_detection_results(vid_writer, save_img, im0s, path, save_dir)

#     finalize_detection(save_txt, save_img, save_dir, vid_writer)

# def handle_vio_detection(vio_pred, vio_img, names, im0s):
#     level_sys = 0 
#     violence_count = 0
#     vio_results_str = ""

#     for i, det in enumerate(vio_pred):  
#         if len(det):
#             img_height, img_width = vio_img.shape[-2:]

#             orig_height, orig_width = im0s.shape[:2] if not isinstance(im0s, list) else im0s[0].shape[:2]

#             det[:, :4] = scale_coords((img_height, img_width), det[:, :4], (orig_height, orig_width)).round()
            
#             for *xyxy, conf, cls in reversed(det):
#                 label_name = names[int(cls)]
#                 conf_value = conf.item()
#                 level_sys = get_level_system(conf_value)

#                 if label_name == 'bicycle':
#                     violence_count += 1
#                     label = f'Violence {conf:.2f}'
#                     plot_one_box(xyxy, im0s, label=label, color=[0, 0, 255], line_thickness=2)
#                     vio_results_str = f"Dangerous level {level_sys}, {violence_count} Violence, "
#         else:
#             vio_results_str = f"Dangerous level {level_sys}, 0 Violence, "

#     return level_sys, violence_count, vio_results_str

# def get_level_system(conf_value):
#     if 0.6 <= conf_value < 0.65:
#         return 1
#     elif 0.65 <= conf_value < 0.75:
#         return 2
#     elif 0.75 <= conf_value:
#         return 3
#     return 0

# def prepare_image(img, device, half):
#     img = torch.from_numpy(img).to(device)
#     img = img.half() if half else img.float()  
#     img /= 255.0  
#     if img.ndimension() == 3:
#         img = img.unsqueeze(0)
#     return img

# def handle_object_detection(pred, resized_im0s, pose_pred, names, im0s):
#     obj_results_str = ""
#     for i, det in enumerate(pred):  
#         p, s, im0, frame = (path[i], '%g: ' % i, resized_im0s[i].copy(), dataset.count) if webcam else (path, '', resized_im0s, getattr(dataset, 'frame', 0))

#         save_path = str(save_dir / Path(p).name)

#         if len(det):
#             det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
#             for *xyxy, conf, cls in reversed(det):
#                 label = f'{names[int(cls)]} {conf:.2f}'
#                 color = [241, 218, 125] if names[int(cls)] == 'person' else colors[int(cls)]
#                 plot_one_box(xyxy, im0, label=label, color=color, line_thickness=1)
#             obj_results_str += f"{len(det)} {names[int(cls)]}{'s' * (len(det) > 1)}, "

#             for idx in range(pose_pred.shape[0]):
#                 plot_skeleton_kpts(im0, pose_pred[idx, 7:].T, 3)

#     return obj_results_str

# def print_detection_summary(vio_results_str, obj_results_str):
#     formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     print(
#         f'\nDetected time:
import argparse
import time
import cv2
import torch
import torch.backends.cudnn as cudnn
from datetime import datetime
from pathlib import Path
from numpy import random
from torchvision import transforms
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, strip_optimizer, set_logging, increment_path, non_max_suppression_kpt
from utils.plots import plot_one_box, output_to_keypoint, plot_skeleton_kpts
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import numpy as np
from keras.models import load_model
from collections import deque
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

# Display the video feed in a separate window
def display_window(img):
    cv2.imshow("Detection", img)
    if cv2.waitKey(1) == ord('q'):  # press 'q' to quit
        return False
    return True

# Load pose model
def load_pose_model(weights_path, device):
    model = torch.load(weights_path, map_location=device)['model'].float().eval()
    model = model.to(device)
    return model

# Pose estimation
def run_pose_estimation(tensor_image, model):
    tensor_image = tensor_image.half() if next(model.parameters()).dtype == torch.float16 else tensor_image.float()
    pred = model(tensor_image)
    pred = pred[0] if isinstance(pred, tuple) else pred
    return non_max_suppression_kpt(pred, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)

# Load violence detection model
def load_vio_model(weights_path, device):
    model = attempt_load(weights_path, map_location=device)
    model.to(device).eval()
    return model

def run_vio_detection(img, model, device, half=False):
    img = img.to(device)
    img = img.half() if half else img.float()  
    img /= 255.0  
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():  
        with torch.amp.autocast('cuda', enabled=half):
            pred = model(img)[0]

    pred = non_max_suppression(pred, 0.6, opt.iou_thres)
    return pred

# Load Keras violence detection model
def load_keras_model(model_path):
    print("Loading Keras model ...")
    return load_model(model_path)

# Keras violence detection
def run_keras_detection(frame, model, Q):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (128, 128)).astype("float32")
    frame = frame.reshape(128, 128, 3) / 255

    preds = model.predict(np.expand_dims(frame, axis=0))[0]
    Q.append(preds)

    results = np.array(Q).mean(axis=0)
    prob = results[0]
    return prob > 0.40, prob

# Main detection function
def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'

    # Load pose estimation model
    pose_model = load_pose_model('yolov7-w6-pose.pt', device)
    model = attempt_load(weights, map_location=device)  

    # Load violence detection YOLO model
    vio_model = load_vio_model('yolov7_vio_detect_v3.pt', device)

    # Load Keras model
    keras_model = load_keras_model("D:/Desktop_Prev_lappy/major project/major_proj_stage2/modelnew.h5")
    Q = deque(maxlen=128)  # Queue for Keras model prediction averaging
    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half() 

    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        dataset = LoadImages(source, img_size=imgsz)
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    prev_frame_time = 0
    new_frame_time = 0

    # Violence and pose detection loop
    for path, img, im0s, vid_cap in dataset:
        # Run YOLO-based violence detection
        new_frame_time = time.time()  # Capture current time at the start of frame processing

        print("Processing Violence Detection...")
        vio_img = torch.from_numpy(img).to(device)
        vio_pred = run_vio_detection(vio_img, vio_model, device, half=half)
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time  # Update time for next frame
        vio_results_str = ""
        level_sys = 0 
        violence_count = 0
        for i, det in enumerate(vio_pred):  
            if len(det):
                img_height, img_width = vio_img.shape[-2:]

                if isinstance(im0s, list):
                    for im0 in im0s:
                        orig_height, orig_width = im0.shape[:2]
                else:
                    orig_height, orig_width = im0s.shape[:2]

                det[:, :4] = scale_coords((img_height, img_width), det[:, :4], (orig_height, orig_width)).round()
                
                for *xyxy, conf, cls in reversed(det):
                    print(names)
                    label_name = names[int(cls)]
                    conf_value = conf.item()  
                    if 0.6 <= conf_value < 0.65:
                        level_sys = 1
                    elif 0.65 <= conf_value < 0.75:
                        level_sys = 2
                    elif 0.75 <= conf_value:
                        level_sys = 3

                    if label_name == 'bicycle':  
                        violence_count += 1
                        label = f'Violence {conf:.2f}'
                        # if isinstance(im0s, (np.ndarray, list)):
                        #     im0s = np.array(im0s)

                        #     plot_one_box(xyxy, im0s, label=label, color=[0, 0, 255], line_thickness=2)
                        # else:
                            # print(f"im0s type: {type(im0s)}, shape: {im0s.shape if isinstance(im0s, np.ndarray) else 'N/A'}")
                            # print(f"Invalid image format at frame {frame}, skipping drawing rectangle.")

                        # plot_one_box(xyxy, im0s, label=label, color=[0, 0, 255], line_thickness=2)
                        vio_results_str = f"Dangerous level {level_sys}, {violence_count} Violence, "
            else:
                vio_results_str = f"Dangerous level {level_sys}, 0 Violence, "
        if not webcam:
            print("Processing Pose Estimation...")
            resized_im0s = letterbox(im0s, 960, stride=64, auto=True)[0]
            resized_tensor = transforms.ToTensor()(resized_im0s).unsqueeze(0).to(device)
            pose_pred = run_pose_estimation(resized_tensor, pose_model)
            pose_pred = output_to_keypoint(pose_pred)

        else:
            resized_im0s = im0s

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  
        img /= 255.0  
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1 = time_synchronized()
        with torch.no_grad():  
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        if classify:
            pred = apply_classifier(pred, modelc, img, resized_im0s)

        obj_results_str = ""
        for i, det in enumerate(pred):  
            if webcam:  
                p, s, im0, frame = path[i], '%g: ' % i, resized_im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', resized_im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  
            save_path = str(save_dir / p.name)  

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    color = [241, 218, 125] if names[int(cls)] == 'person' else colors[int(cls)]
                    plot_one_box(xyxy, im0, label=label, color=color, line_thickness=1)
                obj_results_str += f"{len(det)} {names[int(cls)]}{'s' * (len(det) > 1)}, "

                if not webcam:
                    for idx in range(pose_pred.shape[0]):
                        plot_skeleton_kpts(im0, pose_pred[idx, 7:].T, 3)

            formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f'\nDetected time: {formatted_time}. {vio_results_str}{obj_results_str}Done. ({t2 - t1:.3f}s)')

        # Resize image for pose estimation & Run pose estimation
        if isinstance(im0s, list):
            im0s = im0s[0]  # Extract the first image if it's a list

        resized_im0s = letterbox(im0s, 960, stride=64, auto=True)[0]
        resized_tensor = transforms.ToTensor()(resized_im0s).unsqueeze(0).to(device)
        pose_pred = run_pose_estimation(resized_tensor, pose_model)
        pose_pred = output_to_keypoint(pose_pred)

        # Run Keras-based violence detection
        violence_detected, prob = run_keras_detection(im0s, keras_model, Q)
        text_color = (0, 0, 255) if violence_detected else (0, 255, 0)
        fps_text = f'FPS: {int(fps)}'

        text = f"Violence: {'Yes' if violence_detected else 'No'} | Probability: {prob:.2f}"
        cv2.putText(im0s, text+fps_text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, text_color, 3)
        print(f"Current FPS is {fps_text}, time is {new_frame_time} {prev_frame_time}")
        # Display window and exit on 'q' key
        if not display_window(im0s):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')  # '0' is for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()

    with torch.no_grad():
        detect()
