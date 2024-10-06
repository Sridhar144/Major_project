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
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, non_max_suppression_kpt
from utils.plots import plot_one_box, output_to_keypoint, plot_skeleton_kpts
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import numpy as np
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
    stride = int(model.stride.max())  
    imgsz = check_img_size(imgsz, s=stride)  

    # Load violence detection model
    vio_model = load_vio_model('yolov7_vio_detect_v3.pt', device)

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half() 

    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    
    for path, img, im0s, vid_cap in dataset:
        # Run Violence Detection
        print("Processing Violence Detection...")
        vio_img = torch.from_numpy(img).to(device)
        vio_pred = run_vio_detection(vio_img, vio_model, device, half=half)

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
                        if isinstance(im0s, (np.ndarray, list)):
                            im0s = np.array(im0s)

                            plot_one_box(xyxy, im0s, label=label, color=[0, 0, 255], line_thickness=2)
                        else:
                            print(f"im0s type: {type(im0s)}, shape: {im0s.shape if isinstance(im0s, np.ndarray) else 'N/A'}")
                            print(f"Invalid image format at frame {frame}, skipping drawing rectangle.")

                        # plot_one_box(xyxy, im0s, label=label, color=[0, 0, 255], line_thickness=2)
                        vio_results_str = f"Dangerous level {level_sys}, {violence_count} Violence, "
            else:
                vio_results_str = f"Dangerous level {level_sys}, 0 Violence, "

        # Resize image for pose estimation & Run pose estimation
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

            if not display_window(im0):
                break

            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()

                        if vid_cap:  # Ensure vid_cap is not None
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps = 30  # Default FPS
                            h, w = im0.shape[:2]  # Get dimensions from the image if vid_cap is None

                        fourcc = 'mp4v'  # Codec for saving the video
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    
                    vid_writer.write(im0)


    if save_txt or save_img:
        print(f"Results saved to {save_dir}")

    if vid_writer:
        vid_writer.release()
    cv2.destroyAllWindows()  
    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source') 
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
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()

    with torch.no_grad():
        detect()



            
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
# from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
#     scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, non_max_suppression_kpt
# from utils.plots import plot_one_box, output_to_keypoint, plot_skeleton_kpts
# import matplotlib.pyplot as plt

# from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


# def display(img):
#     plt.imshow(img[..., ::-1])  # Convert BGR to RGB for correct color rendering
#     plt.axis('off')
#     plt.show()


# # Load pose model
# def load_pose_model(weights_path, device):
#     model = torch.load(weights_path, map_location=device)['model'].float().eval()
#     model = model.to(device)
#     return model


# # Pose estimation
# def run_pose_estimation(tensor_image, model):
#     tensor_image = tensor_image.half() if next(model.parameters()).dtype == torch.float16 else tensor_image.float()
#     pred = model(tensor_image)
#     pred = pred[0] if isinstance(pred, tuple) else pred
#     return non_max_suppression_kpt(pred, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)


# # Load violence detection model
# def load_vio_model(weights_path, device):
#     model = attempt_load(weights_path, map_location=device)  # load FP32 model
#     model.to(device).eval()
#     return model


# def run_vio_detection(img, model, device, half=False):
#     img = img.to(device)
#     img = img.half() if half else img.float()  # uint8 to fp16/32
#     img /= 255.0  # 0 - 255 to 0.0 - 1.0
#     if img.ndimension() == 3:
#         img = img.unsqueeze(0)

#     with torch.no_grad():
#         with torch.amp.autocast('cuda', enabled=half):
#             pred = model(img)[0]

#     pred = non_max_suppression(pred, 0.6, opt.iou_thres)  # Confidence threshold 40% for violence detection
#     return pred


# def detect(save_img=False):
#     source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
#     save_img = not opt.nosave and not source.endswith('.txt')
#     webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))

#     save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
#     (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

#     set_logging()
#     device = select_device(opt.device)
#     half = device.type != 'cpu'
#     cyan = [241, 218, 125]

#     pose_model = load_pose_model('yolov7-w6-pose.pt', device)
#     model = attempt_load(weights, map_location=device)
#     stride = int(model.stride.max())
#     imgsz = check_img_size(imgsz, s=stride)

#     vio_model = load_vio_model('yolov7_vio_detect_v3.pt', device)

#     if trace:
#         model = TracedModel(model, device, opt.img_size)

#     if half:
#         model.half()

#     classify = False
#     if classify:
#         modelc = load_classifier(name='resnet101', n=2)
#         modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

#     vid_path, vid_writer = None, None
#     if webcam:
#         view_img = check_imshow()
#         cudnn.benchmark = True
#         dataset = LoadStreams(source, img_size=imgsz, stride=stride)
#     else:
#         dataset = LoadImages(source, img_size=imgsz, stride=stride)

#     names = model.module.names if hasattr(model, 'module') else model.names
#     colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

#     if device.type != 'cpu':
#         model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

#     old_img_w = old_img_h = imgsz
#     old_img_b = 1

#     t0 = time.time()

#     for path, img, im0s, vid_cap in dataset:
#         vio_img = torch.from_numpy(img).to(device)
#         vio_pred = run_vio_detection(vio_img, vio_model, device, half=half)

#         vio_results_str = ""
#         level_sys = 0
#         violence_count = 0
#         for i, det in enumerate(vio_pred):
#             if len(det):
#                 img_height, img_width = vio_img.shape[-2:]

#                 if isinstance(im0s, list):
#                     for im0 in im0s:
#                         orig_height, orig_width = im0.shape[:2]
#                 else:
#                     orig_height, orig_width = im0s.shape[:2]

#                 det[:, :4] = scale_coords((img_height, img_width), det[:, :4], (orig_height, orig_width)).round()

#                 for *xyxy, conf, cls in reversed(det):
#                     label_name = names[int(cls)]
#                     conf_value = conf.item()
#                     if 0.6 <= conf_value < 0.65:
#                         level_sys = 1
#                     elif 0.65 <= conf_value < 0.75:
#                         level_sys = 2
#                     elif 0.75 <= conf_value:
#                         level_sys = 3

#                     if label_name == 'bicycle':
#                         violence_count += 1
#                         label = f'Violence {conf:.2f}'
#                         plot_one_box(xyxy, im0s, label=label, color=[0, 0, 255], line_thickness=2)
#                         vio_results_str = f"Dangerous level {level_sys}, {violence_count} Violence, "
#             else:
#                 vio_results_str = f"Dangerous level {level_sys}, 0 Violence, "

#         if not webcam:
#             resized_im0s = letterbox(im0s, 960, stride=64, auto=True)[0]
#             resized_tensor = transforms.ToTensor()(resized_im0s).unsqueeze(0).to(device)
#             pose_pred = run_pose_estimation(resized_tensor, pose_model)
#             pose_pred = output_to_keypoint(pose_pred)
#         else:
#             resized_im0s = im0s

#         img = torch.from_numpy(img).to(device)
#         img = img.half() if half else img.float()
#         img /= 255.0
#         if img.ndimension() == 3:
#             img = img.unsqueeze(0)

#         if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
#             old_img_b = img.shape[0]
#             old_img_h = img.shape[2]
#             old_img_w = img.shape[3]
#             for i in range(3):
#                 model(img, augment=opt.augment)[0]

#         t1 = time_synchronized()
#         with torch.no_grad():
#             pred = model(img, augment=opt.augment)[0]
#         t2 = time_synchronized()

#         pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
#         t3 = time_synchronized()

#         obj_results_str = ""
#         for i, det in enumerate(pred):
#             if webcam:
#                 p, s, im0, frame = path[i], '%g: ' % i, resized_im0s[i].copy(), dataset.count
#             else:
#                 p, s, im0, frame = path, '', resized_im0s, getattr(dataset, 'frame', 0)

#             p = Path(p)
#             save_path = str(save_dir / p.name)

#             if len(det):
#                 det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

#                 for *xyxy, conf, cls in reversed(det):
#                     label = f'{names[int(cls)]} {conf:.2f}'
#                     color = cyan if names[int(cls)] == 'person' else colors[int(cls)]
#                     plot_one_box(xyxy, im0, label=label, color=color, line_thickness=1)
#                 obj_results_str += f"{len(det)} {names[int(cls)]}{'s' * (len(det) > 1)}, "

#                 if not webcam:
#                     display(im0)
#                 else:
#                     cv2.imshow(str(p), im0)

#                 if cv2.waitKey(1) == ord('q'):
#                     raise StopIteration

#         if opt.save_img:
#             if dataset.mode == 'images':
#                 cv2.imwrite(save_path, im0)
#             else:
#                 if vid_path != save_path:
#                     vid_path = save_path
#                     if isinstance(vid_writer, cv2.VideoWriter):
#                         vid_writer.release()
#                     fps = vid_cap.get(cv2.CAP_PROP_FPS)
#                     w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#                     h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#                     vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
#                 vid_writer.write(im0)

#     print(f'Done. ({time.time() - t0:.3f}s)')


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
#     parser.add_argument('--source', type=str, default='inference/images', help='source')
#     parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
#     parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
#     parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
#     parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     parser.add_argument('--view-img', action='store_true', help='display results')
#     parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
#     parser.add_argument('--save-conf', action='store_true', help='save confidences in *.txt labels')
#     parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
#     parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
#     parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
#     parser.add_argument('--augment', action='store_true', help='augmented inference')
#     parser.add_argument('--update', action='store_true', help='update all models')
#     parser.add_argument('--project', default='runs/detect', help='save results to project/name')
#     parser.add_argument('--name', default='exp', help='save results to project/name')
#     parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
#     parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
#     parser.add_argument('--save-img', action='store_true', help='save images/videos')
#     opt = parser.parse_args()

#     with torch.no_grad():
#         if opt.update:
#             for opt.weights in ['yolov7.pt']:
#                 detect()
#         else:
#             detect()
