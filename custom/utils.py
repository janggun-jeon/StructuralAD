import os
import torch
import torch.nn.functional as F
import numpy as np
import math
import cv2

from PIL import Image

from ultralytics import YOLO
from preprocessing.single_image_enhance_tflite import zeroDCE

def roi_selection(img, x, y, w, h):
    patch = img[:, y:h, x:w]
    roi = F.interpolate(patch.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
    del patch
    return roi
    
def roi_upper(img):
    roi = roi_selection(img, 1172, 40, 1772, 640)
    del img
    return roi

def roi_left(img):
    roi = roi_selection(img, 336, 776, 936, 1376)
    del img
    return roi

def roi_lower(img):
    roi = roi_selection(img, 1272, 1512, 1872, 2112)
    del img
    return roi

def roi_right(img):
    roi = roi_selection(img, 2108, 776, 2708, 1376)
    del img
    return roi

class ROI:
    def __init__(self, func):
        self.func = func
        
    def __call__(self, img):
        return self.func(img)

'''============================================================================================================'''

distance_zero_count = 0

def zoom(img, masks, target_distance=1200, bg_color=(255, 255, 255)):
    global distance_zero_count

    def adjust_image(img, masks, target_distance, bg_color):
        global distance_zero_count
        
        def euclidean_distance(point1, point2):
            return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
        
        if not (masks and hasattr(masks, 'xy')):
            distance_zero_count += 1
            img_center = (img.shape[1] // 2, img.shape[0] // 2)
            return img, img_center, 0

        centers = [(mask[:, 0].mean(), mask[:, 1].mean()) for mask in masks.xy if len(mask) > 1]
        img_center = (img.shape[1] // 2, img.shape[0] // 2)

        if len(centers) < 2:
            distance_zero_count += 1
            return img, img_center, 0

        distances_from_center = sorted([(math.sqrt((img_center[0] - x) ** 2 + (img_center[1] - y) ** 2), (x, y)) for x, y in centers])

        filtered_centers = []
        for i in range(len(distances_from_center)):
            for j in range(i + 1, len(distances_from_center)):
                distance_between_centers = euclidean_distance(distances_from_center[i][1], distances_from_center[j][1])
                if distance_between_centers > 200:
                    filtered_centers.append(distances_from_center[i][1])
                    filtered_centers.append(distances_from_center[j][1])
                    break
            if len(filtered_centers) >= 2:
                break

        if len(filtered_centers) < 2:
            distance_zero_count += 1
            return img, img_center, 0

        closest_centers = filtered_centers[:2]
        distance = euclidean_distance(*closest_centers)
        higher_point, lower_point = max(closest_centers, key=lambda p: p[1]), min(closest_centers, key=lambda p: p[1])
        intersection_point = (int(lower_point[0]), int(higher_point[1]))

        dx, dy = img_center[0] - intersection_point[0], img_center[1] - intersection_point[1]
        moved_img = cv2.warpAffine(img, np.float32([[1, 0, dx], [0, 1, dy]]), (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=bg_color)

        if distance == 0:
            return moved_img, img_center, 0
        
        scale_factor = target_distance / distance
        new_size = (int(moved_img.shape[1] * scale_factor), int(moved_img.shape[0] * scale_factor))
        resized_img = cv2.resize(moved_img, new_size, interpolation=cv2.INTER_LINEAR)

        if scale_factor > 1:  # 확대
            x_offset, y_offset = (resized_img.shape[1] - moved_img.shape[1]) // 2, (resized_img.shape[0] - moved_img.shape[0]) // 2
            cropped_img = resized_img[y_offset:y_offset + moved_img.shape[0], x_offset:x_offset + moved_img.shape[1]]
            return cropped_img, intersection_point, distance
        else:  # 축소
            canvas = np.full_like(moved_img, bg_color, dtype=np.uint8)
            x_offset, y_offset = (moved_img.shape[1] - resized_img.shape[1]) // 2, (moved_img.shape[0] - resized_img.shape[0]) // 2
            canvas[y_offset:y_offset + resized_img.shape[0], x_offset:x_offset + resized_img.shape[1]] = resized_img
            return canvas, intersection_point, distance

    adjusted_img, _, _ = adjust_image(img, masks, target_distance, bg_color)
    return adjusted_img

'''============================================================================================================'''

def filter2D(img):
    kernel = np.ones((3,3), dtype=np.float64) / 9. 
    
    return cv2.filter2D(img, -1, kernel)

def canny(img):
    '''
    img, threshold1, threshold2
    img : 입력 이미지, 보통 그레이스케일 이미지 사용
    threshold1 : 하위 임계값, 이 값보다 작을 경우 무시됨
    threshold2 : 상위 임계값, 이 값보다 클 경우 확실한 edge로 간주
    '''
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 채널 단일화
    
    return cv2.Canny(gray, 28, 37)

def merge(origin, modified):
    background = Image.fromarray(origin)
    foreground = Image.fromarray(modified)
    background.paste(foreground, (0,0), foreground)

    return np.array(background)

class Enhancing:
    def __init__(self, segmenter_path='pre_trained/detector/weights/yolov8s-seg.pt'):
        self.segmenter_path = segmenter_path # 'results/detector/weights/yolov8s-seg.pt'
        
    def __call__(self, img):
        img = zeroDCE(img_info=img).numpy()
        img = filter2D(img)

        canny_img = None
        merge_img = None

        canny_img = canny(img)
        merge_img = merge(img, canny_img)
        
        segmenter = YOLO(os.path.join(os.getcwd(), self.segmenter_path)) #.to('cpu') 
        [_, _, masks, _] = segmenter.predict(source=merge_img, save=True, show_boxes=False, show_labels=False, show_conf=False, imgsz=1312, conf=0.6, retina_masks=False) #, device='cpu')
        
        torch.cuda.empty_cache()
        
        del segmenter
        
        enhanced_img = zoom(merge_img, masks=masks)
        
        return enhanced_img.transpose((2, 0, 1))