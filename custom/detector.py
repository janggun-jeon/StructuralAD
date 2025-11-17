import cv2
import torch
import numpy as np
from scipy.spatial import distance
import math
import os
import re
import time

from torch.utils.data import Dataset

from ultralytics import YOLO
from preprocessing.single_image_enhance_tflite import zeroDCE
from custom.utils import filter2D, canny, merge, zoom

def Hough_Transform(img):
    gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray1, (9, 9), 2)
    
    """
    image, method, dp
    4) minDist : 검출된 원들 간의 최소 거리, 검출할 원의 중심 사이의 최소 거리로, 값이 작을수록 원이 더 많이 검출됨
    5) param1 : Canny edge 검출기의 상위 임계값
    6) param2 : 색상 민감도, 숫자가 작을수록 민감도가 떨어짐
    7) minRadius : 검출할 원의 최소 반지름
    8) maxRadius : 검출할 원의 최대 반지름
    """
   
    return cv2.HoughCircles(gray2, cv2.HOUGH_GRADIENT, 1, minDist=26, param1=13, param2=19, minRadius=35, maxRadius=56) 
    # return cv2.HoughCircles(gray2, cv2.HOUGH_GRADIENT, 1, minDist=25, param1=13, param2=19, minRadius=35, maxRadius=56) 

def detect(img, method):
        return method(img)

def xyxy2xy(xyxy):
    x = xyxy[:,0] + xyxy[:,2]    
    y = xyxy[:,1] + xyxy[:,3]
    return torch.cat((y.unsqueeze(1), x.unsqueeze(1)), dim=1) / 2

"""
이미지 시각화 -> 원탐지 탐지 x이거나 roi 이상한 것들만 시각화하는 코드로 변경 필요
"""
def visualize_result(img, patch, x, y, w, h, normality):
    for itr in patch:
        cv2.circle(img, (itr[0], itr[1]), 1, (0, 100, 100), 3)
        cv2.circle(img, (itr[0], itr[1]), itr[2], (255, 0, 255), 2)

    if normality == 1:
        line_color = (0, 255, 0)  
        line_thickness = 2
    else:
        line_color = (0, 0, 255) 
        line_thickness = 2
    
    cv2.line(img, (x, y), (x, h), line_color, line_thickness)
    cv2.line(img, (x, y), (w, y), line_color, line_thickness)
    cv2.line(img, (w, y), (w, h), line_color, line_thickness)
    cv2.line(img, (x, h), (w, h), line_color, line_thickness)

"""
12시 방향 range에 사용됨
"""
def range_af_2(AF, normality, patch1, masks):
    Vertical_Length = []
    threshold = 92
    mask_array = masks.data.cpu().numpy()
    for i, mask in enumerate(mask_array):
        if len(mask.shape) > 2:
            mask = mask.squeeze()
        vertical_length = np.sum(mask, axis=0).max()
        Vertical_Length.append(vertical_length)

    is_seg = False
    for a in Vertical_Length:
        if a > threshold:
            is_seg = True
            idx = 0
            if ('-AF_2,' in AF):
                idx = AF.index('-AF_2,') - 1
            elif ('-AF_2s,' in AF):
                idx = AF.index('-AF_2s,') - 1

            if AF[idx] != '1':
                AF[idx] = '1'

            if (len(patch1) + int(AF[idx]) * 3) != 5:
                normality = 0
                break

    if not is_seg:
        normality = 0

    return normality

"""
12시 방향 range
"""
def range_12(img, points, roi, AF, masks, normality):
    tmp = normality
    normality = 1
    
    """
    ㄷ자 mid 방향 처리
    """
    x_mid = np.int16(np.around(roi[1] + 220))
    y_mid = np.int16(np.around(roi[0] - 1264))
    w_mid = np.int16(np.around(roi[1] + 550))
    h_mid = np.int16(np.around(roi[0] - 750))

    x_AF = np.int16(np.around(roi[1] + 220))
    y_AF = np.int16(np.around(roi[0] - 1200))
    w_AF = np.int16(np.around(roi[1] + 310))
    h_AF = np.int16(np.around(roi[0] - 937))
    
    # 12시 방향 범위 내의 점들 선택
    patch_idx = np.where(
        (points[:, 0] > x_mid) &\
        (points[:, 0] < w_mid) &\
        (points[:, 1] > y_mid) &\
        (points[:, 1] < h_mid)
    )
    patch_mid = points[patch_idx[0]]
    
    patch_idx = np.where(
        (patch_mid[:, 0] <= x_AF) |\
        (patch_mid[:, 0] >= w_AF) |\
        (patch_mid[:, 1] <= y_AF) |\
        (patch_mid[:, 1] >= h_AF)
    )
    patch_mid = patch_mid[patch_idx[0]]

    # 색상 조건에 맞는 점들만 선택
    patch_mid = [circle for circle in patch_mid if all(color <= 94 for color in img[circle[1], circle[0]])]

    # AF 세그멘테이션 탐지 및 처리
    if ('-AF_2,' in AF) or ('-AF_2s,' in AF):
        normality = range_af_2(AF, normality, patch_mid, masks)
    else:
        normality = 0

    # 12시 방향 결과 시각화
    visualize_result(img, patch_mid, x_mid, y_mid, w_mid, h_mid, normality)
  
    return np.min([normality, tmp])

"""
9시 방향 left range에 사용됨
"""
def range_af_1(AF, normality, masks):
    Vertical_Length1 = []
    low_threshold = 52
    high_threshold = 116
    mask_array = masks.data.cpu().numpy()
    for i, mask in enumerate(mask_array):
        if len(mask.shape) > 2:
            mask = mask.squeeze()
        vertical_length1 = np.sum(mask, axis=0).max()
        Vertical_Length1.append(vertical_length1)
    is_seg = True
    for a in Vertical_Length1:
        if a < low_threshold:
            is_seg = False
            
        elif a > high_threshold:
            is_seg = False
    idx = 0 
    if ('-AF_1,' in AF):
        idx = AF.index('-AF_1,') - 1
    elif ('-AF_1s,' in AF):
        idx = AF.index('-AF_1s,') - 1

    if AF[idx] != '1':
        AF[idx] = '1'

    if (int(AF[idx]) * 2) != 2:
        normality = 0

    if not is_seg:
        normality = 0  
    
    return normality

"""
9시 방향 range
"""
def range_9(img, points, roi, AF, masks, normality):
    tmp = normality
    normality = 1   
    
    """
    Right 부분 처리
    """
    x_right = np.int16(np.around(roi[1] - 820))
    y_right = np.int16(np.around(roi[0] - 415))# 405)) # y < 402
    w_right = np.int16(np.around(roi[1] - 280))
    h_right = np.int16(np.around(roi[0] - 174))

    x_AF = np.int16(np.around(roi[1] - 640))
    y_AF = np.int16(np.around(roi[0] - 377))
    w_AF = np.int16(np.around(roi[1] - 560))
    h_AF = np.int16(np.around(roi[0] - 267))    
    
    patch_idx = np.where(
        (points[:, 0] > x_right) &\
        (points[:, 0] < w_right) &\
        (points[:, 1] > y_right) &\
        (points[:, 1] < h_right)
    )
    patch_right = points[patch_idx[0]]
    
    patch_idx = np.where(
        (patch_right[:, 0] <= x_AF) |\
        (patch_right[:, 0] >= w_AF) |\
        (patch_right[:, 1] <= y_AF) |\
        (patch_right[:, 1] >= h_AF)
    )
    patch_right = patch_right[patch_idx[0]]    
       
    patch_right = [circle for circle in patch_right if all(color < 83 for color in img[circle[1], circle[0]])]
    if len(patch_right) != 2:
        normality = 0
    else:
        if ('-AF_1,' in AF) or ('-AF_1s,' in AF):
            normality = range_af_1(AF, normality, masks)
        else:
            normality = 0

    # 9시 방향 right 결과 시각화
    visualize_result(img, patch_right, x_right, y_right, w_right, h_right, normality)

    return np.min([normality, tmp])

"""
6시 방향 range
"""
def range_6(img, points, roi, normality):
    tmp = normality
    normality = 1   
    
    x = np.int16(np.around(roi[1] + 330))
    y = np.int16(np.around(roi[0] + 350))
    w = np.int16(np.around(roi[1] + 645)) # 602 < w
    h = np.int16(np.around(roi[0] + 606))
    
    x_2 = np.int16(np.around(roi[1] + 624))
    y_2 = np.int16(np.around(roi[0] + 350))
    w_2 = np.int16(np.around(roi[1] + 645))
    h_2 = np.int16(np.around(roi[0] + 422))#425))

    x_3 = np.int16(np.around(roi[1] + 631))
    y_3 = np.int16(np.around(roi[0] + 470))
    w_3 = np.int16(np.around(roi[1] + 633))
    h_3 = np.int16(np.around(roi[0] + 472))
    
    x_AF = np.int16(np.around(roi[1] + 330))
    y_AF = np.int16(np.around(roi[0] + 350))
    w_AF = np.int16(np.around(roi[1] + 360))
    h_AF = np.int16(np.around(roi[0] + 400))      
    
    patch_idx = np.where(
        (points[:, 0] > x) &\
        (points[:, 0] < w) &\
        (points[:, 1] > y) &\
        (points[:, 1] < h)
    )
    patch3 = points[patch_idx[0]]
    # visualize_result2(img, patch3)
    patch_idx = np.where(
        (patch3[:, 0] <= x_2) |\
        (patch3[:, 0] >= w_2) |\
        (patch3[:, 1] <= y_2) |\
        (patch3[:, 1] >= h_2)
    )
    patch3 = patch3[patch_idx[0]]  
    
    patch_idx = np.where(
        (patch3[:, 0] <= x_3) |\
        (patch3[:, 0] >= w_3) |\
        (patch3[:, 1] <= y_3) |\
        (patch3[:, 1] >= h_3)
    )
    patch3 = patch3[patch_idx[0]]  
    
    patch_idx = np.where(
        (patch3[:, 0] <= x_AF) |\
        (patch3[:, 0] >= w_AF) |\
        (patch3[:, 1] <= y_AF) |\
        (patch3[:, 1] >= h_AF)
    )
    patch3 = patch3[patch_idx[0]]    
    
    patch3 = [circle for circle in patch3 if all(color < 100 for color in img[circle[1], circle[0]])]
    
    # 유클리드 거리가 120 미만인 점들만 남기기
    patch3 = np.array(patch3)
    if len(patch3) > 1:
        dists = distance.cdist(patch3, patch3, 'euclidean')
        close_points = np.any((dists < 94) & (dists != 0), axis=1) # close_points = np.any((dists < 91) & (dists != 0), axis=1)
        patch3 = patch3[close_points]
    
    if len(patch3) != 2:
        normality = 0
    
    visualize_result(img, patch3, x, y, w, h, normality)

    return np.min([normality, tmp])

"""
3시 방향 range
"""
def range_3(img, points, roi, normality):
    tmp = normality
    normality = 1   
    
    x = np.int16(np.around(roi[1] + 1100))
    y = np.int16(np.around(roi[0] - 450))
    w = np.int16(np.around(roi[1] + 1500))
    h = np.int16(np.around(roi[0] - 151)) #150 <=h
    
    x_AF = np.int16(np.around(roi[1] + 1207))
    y_AF = np.int16(np.around(roi[0] - 156))
    w_AF = np.int16(np.around(roi[1] + 1209))
    h_AF = np.int16(np.around(roi[0] - 151))  
    
    patch_idx = np.where(
        (points[:, 0] > x) &\
        (points[:, 0] < w) &\
        (points[:, 1] > y) &\
        (points[:, 1] < h)
    )
    patch4 = points[patch_idx[0]]
   
    patch_idx = np.where(
        (patch4[:, 0] <= x_AF) |\
        (patch4[:, 0] >= w_AF) |\
        (patch4[:, 1] <= y_AF) |\
        (patch4[:, 1] >= h_AF)
    )
    patch4 = patch4[patch_idx[0]] 
    
    patch4 = [circle for circle in patch4 if all(color < 92 for color in img[circle[1], circle[0]])] # 92

    # 유클리드 거리가 120 미만인 점들만 남기기
    patch4 = np.array(patch4)
    if len(patch4) > 1:
        dists = distance.cdist(patch4, patch4, 'euclidean')
        close_points = np.any((dists < 140) & (dists != 0), axis=1)
        patch4 = patch4[close_points]

    if len(patch4) != 2:
        normality = 0
        
    visualize_result(img, patch4, x, y, w, h, normality)

    return np.min([normality, tmp])

''' 메모리 사용량 줄이기 '''
def ROI_Selection(img, points, roi, AF, masks, normality):
    normality = range_12(img, points, roi, AF, masks, normality)
    normality = range_9(img, points, roi, AF, masks, normality)
    normality = range_6(img, points, roi, normality)
    normality = range_3(img, points, roi, normality)
    
    del points
    
    return normality

def natural_sort_key(string):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', string)]

class Image_Loader(Dataset):
    def __init__(self, dir_path):
        imgs_paths = np.array([os.path.join(dir_path, img_path) for img_path in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, img_path))])
        
        self.imgs_paths = np.array(sorted(imgs_paths, key=natural_sort_key))
        self.imgs_paths = self.imgs_paths[0:]

    def __len__(self):
        return len(self.imgs_paths)
        
    def __getitem__(self, index):
        img_path = self.imgs_paths[index]
        original_filename = os.path.basename(img_path)
        
        img = cv2.imread(img_path)
        img = zeroDCE(img_info=img_path).numpy()
        img = filter2D(img)

        canny_img = None
        merge_img = None

        canny_img = canny(img)
        merge_img = merge(img, canny_img)

        img = merge_img
        
        return img, original_filename

def AD(img, name, segmenter):
    torch.cuda.empty_cache()
    [classes, img, masks, _] = segmenter.predict(source=img, save=True, show_boxes=False, show_labels=False, show_conf=False, imgsz=1312, conf=0.6, retina_masks=False)
    torch.cuda.empty_cache()
    
    normality = 1
    roi = None
    if masks is None:
        normality = 0
    else:
        try:
            centers = [(mask[:, 0].mean(), mask[:, 1].mean()) for mask in masks.xy if len(mask) > 0 and len(mask[0]) > 1]
            img_center = (img.shape[1] // 2, img.shape[0] // 2)

            distances_from_center = sorted([(math.sqrt((img_center[0] - x) ** 2 + (img_center[1] - y) ** 2), (x, y)) for x, y in centers])

            filtered_centers = []
            for i in range(len(distances_from_center)):
                for j in range(i + 1, len(distances_from_center)):
                    distance_between_centers = math.sqrt((distances_from_center[i][1][0] - distances_from_center[j][1][0]) ** 2 + (distances_from_center[i][1][1] - distances_from_center[j][1][1]) ** 2)
                    if distance_between_centers > 200:
                        filtered_centers.append(distances_from_center[i][1])
                        filtered_centers.append(distances_from_center[j][1])
                        break
                if len(filtered_centers) >= 2:
                    break

            closest_centers = filtered_centers[:2]

            if len(closest_centers) > 1:
                higher_point = max(closest_centers, key=lambda p: p[1])
                lower_point = min(closest_centers, key=lambda p: p[1])
                intersection_point = (lower_point[0], higher_point[1])

                roi = (int(intersection_point[0]), int(intersection_point[1]))

        except IndexError:
            roi = None

    info = detect(img, method=Hough_Transform)
    if info is not None and info.size > 0:
        info = np.uint16(np.around(info))
    else:
        pass

    if roi == None:
        normality = 0
    else:
        points = np.array(info[0][:, :3])
        normality = ROI_Selection(img, points, roi, classes, masks, normality)

    if int(normality) > 0:
        os.makedirs(os.path.join(os.getcwd(), 'results/Detector/images/good'), exist_ok=True)
        cv2.imwrite(os.path.join(os.getcwd(), 'results/Detector/images/good', name), img)
    else:
        os.makedirs(os.path.join(os.getcwd(), 'results/Detector/images/anomaly'), exist_ok=True)
        cv2.imwrite(os.path.join(os.getcwd(), 'results/Detector/images/anomaly', name), img)
    
    del img
    del info
    del roi
    del classes
    del masks

    return normality

def predict(source='datasets/BALL/ball'):
    path = os.path.join(os.getcwd(), source)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    segmenter = YOLO(os.path.join(os.getcwd(), 'pre_trained/detector/weights/yolov8s-seg.pt'))
        
    results = {'TP': 0, 'FN': 0, 'TN': 0, 'FP': 0}
    keeped_times = time.time() 

    img_loader = Image_Loader(dir_path=os.path.join(path, 'train/good'))
    for itr, (img, name) in enumerate(img_loader):
        
        [_, _, masks, _] = segmenter.predict(source=img, save=True, show_boxes=True, show_labels=False, show_conf=False, imgsz=1312, conf=0.6, retina_masks=False)
        img = zoom(img, masks=masks)
        results['TN'] += AD(img, name=name, segmenter=segmenter)

        print(f'\r진행률(train/good) : {itr+1}/{len(img_loader)}', end='', flush=True)
    print()
    results['FP'] += len(img_loader)
    del img_loader
        
    img_loader = Image_Loader(dir_path=os.path.join(path, 'test/good'))
    for itr, (img, name) in enumerate(img_loader):
        [_, _, masks, _] = segmenter.predict(source=img, save=True, show_boxes=True, show_labels=False, show_conf=False, imgsz=1312, conf=0.6, retina_masks=False)
        img = zoom(img, masks=masks)
        results['TN'] += AD(img, name=name, segmenter=segmenter)
        
        print(f'\r진행률(test/good) : {itr+1}/{len(img_loader)}', end='', flush=True)
    print()    
    results['FP'] += (len(img_loader) - results['TN'])
    del img_loader

    img_loader = Image_Loader(dir_path=os.path.join(path, 'test/anomaly'))
    for itr, (img, name) in enumerate(img_loader):
        [_, _, masks, _] = segmenter.predict(source=img, save=True, show_boxes=True, show_labels=False, show_conf=False, imgsz=1312, conf=0.6, retina_masks=False)
        img = zoom(img, masks=masks)
        results['FN'] += AD(img, name=name, segmenter=segmenter)
        
        print(f'\r진행률(test/anomaly) : {itr+1}/{len(img_loader)}', end='', flush=True)
    print()    
    results['TP'] += (len(img_loader) - results['FN'])
    del img_loader 

    return results, time.time() - keeped_times
 
def report(results, keeped_times):
        TP = results['TP']
        TN = results['TN']
        FP = results['FP']
        FN = results['FN']

        # Precision, Recall, F1 계산
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        support = TP + TN + FP + FN  # 전체 데이터 수

        # 레이아웃 출력
        print("Classification Report:")
        print()
        print(f"{'':<14}{'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}")
        print()
        print(f"{'Overall':<14}{precision:>10.4f} {recall:>10.4f} {f1_score:>10.4f} {support:>10}")
        print()
        print(f'Average Times: {keeped_times/support:,.4f}sec') 