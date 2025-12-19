import cv2
import numpy as np

# 1. 初始化資料
img_target = cv2.imread('target.jpg') 
if img_target is None:
    print("找不到圖片，請檢查檔名是否正確")
    exit()

img_show = img_target.copy()
h, w = img_target.shape[:2]
mask = np.zeros((h, w), np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# 2. 第一階段：大範圍框選 (初始化)
cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Select ROI", 600, 800)
rect = cv2.selectROI("Select ROI", img_target, False)
cv2.destroyWindow("Select ROI")

if rect[2] > 0 and rect[3] > 0:
    # 執行初步 GrabCut
    print("正在執行初步去背...")
    cv2.grabCut(img_target, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    
    # 建立視窗顯示當前結果
    win_name = 'Refine Mode (Right Click to Box Shadow)'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 600, 800)

    # 定義滑鼠事件：只處理右鍵框選
    def on_mouse(event, x, y, flags, param):
        global mask, img_show
        if event == cv2.EVENT_RBUTTONDOWN:
            print("請框選消失的手臂區域，然後按 Enter")
            # 在目前的視窗再次進行局部框選
            sub_rect = cv2.selectROI(win_name, img_show, False)
            if sub_rect[2] > 0 and sub_rect[3] > 0:
                sx, sy, sw, sh = sub_rect
                # 將該區域強制標記為「確定前景 (GC_FGD)」
                mask[sy:sy+sh, sx:sx+sw] = cv2.GC_FGD
                # 在畫面上畫個綠框標示已選取
                cv2.rectangle(img_show, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 3)
                print("局部區域已標記，請按 'n' 重新計算")

    cv2.setMouseCallback(win_name, on_mouse)

    while True:
        # 顯示目前的圖 (包含你畫的綠色小框)
        cv2.imshow(win_name, img_show)
        
        # 同時顯示去背後的結果對照
        mask_tmp = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        res = img_target * mask_tmp[:, :, np.newaxis]
        cv2.namedWindow('Current Result', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Current Result', 600, 800)
        cv2.imshow('Current Result', res)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('n'): # 按 n 重新計算
            print("正在根據局部框選重新計算...")
            cv2.grabCut(img_target, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
        elif key == ord('q'): # 按 q 退出
            break

    # 最後處理
    mask_final = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
    mask_final = cv2.GaussianBlur(mask_final, (5, 5), 0)
    cv2.imwrite('final_mask.png', mask_final)

    cv2.destroyAllWindows()