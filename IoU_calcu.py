import sys
import cv2
import numpy as np
import os
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox

def calculate_iou_with_dialog():
    # 初始化 PyQt 應用程式 (彈出視窗需要)
    app = QApplication(sys.argv)

    # 1. 選擇 Ground Truth Mask (SAM)
    gt_path, _ = QFileDialog.getOpenFileName(None, "選擇 SAM 生成的 Ground Truth (GT) Mask", "", "Images (*.png *.jpg *.jpeg)")
    if not gt_path:
        return

    # 2. 選擇測試 Mask (來自你的 test_ui.py)
    test_path, _ = QFileDialog.getOpenFileName(None, "選擇你工具產生的 Mask", "", "Images (*.png *.jpg *.jpeg)")
    if not test_path:
        return

    # 讀取影像
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    test = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)

    if gt is None or test is None:
        QMessageBox.critical(None, "錯誤", "無法讀取影像檔案！")
        return

    # 尺寸對齊 (SAM 的輸出可能與你的原圖尺寸不同)
    if gt.shape != test.shape:
        test = cv2.resize(test, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 二值化 (0 或 1)
    _, gt_bin = cv2.threshold(gt, 127, 1, cv2.THRESH_BINARY)
    _, test_bin = cv2.threshold(test, 127, 1, cv2.THRESH_BINARY)

    # 計算 IoU
    intersection = np.logical_and(gt_bin, test_bin)
    union = np.logical_or(gt_bin, test_bin)
    
    iou_score = np.sum(intersection) / np.sum(union)

    # 視覺化比較圖
    h, w = gt.shape
    diff_img = np.zeros((h, w, 3), dtype=np.uint8)
    # 綠色：兩者重合 (正確)
    diff_img[intersection] = [0, 255, 0] 
    # 藍色：SAM 有，你沒有 (漏抓)
    diff_img[np.logical_and(gt_bin == 1, test_bin == 0)] = [255, 0, 0]
    # 紅色：SAM 沒有，你有 (多抓)
    diff_img[np.logical_and(gt_bin == 0, test_bin == 1)] = [0, 0, 255]

    # 顯示結果視窗
    result_text = f"IoU Score: {iou_score:.4f}\n準確率: {iou_score * 100:.2f}%"
    QMessageBox.information(None, "計算結果", result_text)

    # 顯示比較圖
    cv2.imshow(f"Green: Correct | Blue: Missing | Red: Extra (IoU: {iou_score:.4f})", diff_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    calculate_iou_with_dialog()