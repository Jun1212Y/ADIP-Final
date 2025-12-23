import cv2
import numpy as np

def process_and_grabcut(image_path):
    # 1. 讀取原始圖片
    img = cv2.imread(image_path)
    if img is None:
        print(f"找不到檔案: {image_path}")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, crack_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    crack_mask = cv2.dilate(crack_mask, kernel, iterations=1)
    img_fixed = cv2.inpaint(img, crack_mask, 3, cv2.INPAINT_TELEA)
    lab = cv2.cvtColor(img_fixed, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    img_enhanced = cv2.merge((cl, a, b))
    img_enhanced = cv2.cvtColor(img_enhanced, cv2.COLOR_LAB2BGR)
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (5, 5, w-10, h-10)
    cv2.grabCut(img_enhanced, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    grabcut_res = img_enhanced * mask2[:, :, np.newaxis]
    crack_visual = cv2.cvtColor(crack_mask, cv2.COLOR_GRAY2BGR) # 轉三通道才能合併
    
    # 水平合併：原始圖 | 偵測裂痕 | 修復後 | 對比增強後 | GrabCut去背
    # 因為圖太多，建議分兩組顯示或縮放
    row1 = np.hstack((img, crack_visual, img_fixed))
    row2 = np.hstack((img_enhanced, grabcut_res, np.zeros_like(img))) # 第三格補黑塊
    
    final_stack = np.vstack((row1, row2))
    
    # 縮放以便螢幕觀看
    scale = 0.4
    display_res = cv2.resize(final_stack, None, fx=scale, fy=scale)

    cv2.imshow('Top(L-R): Org, CrackMask, Inpainted | Bottom: Enhanced, Final Result', display_res)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 執行
process_and_grabcut('old_photo_dataset/09/00043.png')