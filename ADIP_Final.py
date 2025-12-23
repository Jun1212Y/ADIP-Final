# import cv2
# import numpy as np

# # ===============================
# # Config
# # ===============================
# ITER_INIT = 5
# ITER_LOCAL = 5
# ITER_GLOBAL = 5
# DISPLAY_SIZE = (600, 800)

# # ===============================
# # Display helper (keep aspect ratio)
# # ===============================
# def resize_window_keep_aspect(win_name, img, max_w=1200, max_h=800):
#     h, w = img.shape[:2]
#     scale = min(max_w / w, max_h / h, 1.0)
#     cv2.resizeWindow(win_name, int(w * scale), int(h * scale))

# # ===============================
# # Helper functions
# # ===============================
# def select_roi_or_skip(win, img, msg):
#     print(msg)
#     cv2.namedWindow(win, cv2.WINDOW_NORMAL)
#     resize_window_keep_aspect(win, img)
#     rect = cv2.selectROI(win, img, fromCenter=False, showCrosshair=False)
#     cv2.destroyWindow(win)

#     x, y, w, h = rect
#     if w <= 0 or h <= 0:
#         return None

#     H, W = img.shape[:2]
#     x = max(0, x); y = max(0, y)
#     w = min(w, W - x); h = min(h, H - y)
#     if w <= 0 or h <= 0:
#         return None

#     cv2.destroyAllWindows()
#     return (x, y, w, h)

# def show_current(img, mask, title="Current Result"):
#     fg = np.where(
#         (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
#         255, 0
#     ).astype(np.uint8)

#     result = cv2.bitwise_and(img, img, mask=fg)

#     cv2.namedWindow(title, cv2.WINDOW_NORMAL)
#     cv2.imshow(title, result)
#     resize_window_keep_aspect(title, result)
#     cv2.waitKey(1)

# # ===============================
# # 1. Read image
# # ===============================
# img = cv2.imread("target.jpg")
# if img is None:
#     raise SystemExit("Cannot find target.jpg")

# h, w = img.shape[:2]

# # ===============================
# # 2. Select PERSON ROI
# # ===============================
# person_roi = select_roi_or_skip(
#     "Select PERSON ROI",
#     img,
#     "Step 1: Select the ROI"
# )
# if person_roi is None:
#     raise SystemExit("No ROI selected")

# px, py, pw, ph = person_roi

# # ===============================
# # 3. Initial GrabCut
# # ===============================
# mask = np.zeros((h, w), np.uint8)
# bgdModel = np.zeros((1, 65), np.float64)
# fgdModel = np.zeros((1, 65), np.float64)

# cv2.grabCut(
#     img, mask, (px, py, pw, ph),
#     bgdModel, fgdModel,
#     ITER_INIT,
#     cv2.GC_INIT_WITH_RECT
# )

# show_current(img, mask, "After Initial GrabCut")

# # ===============================
# # 4. Interactive refinement
# # ===============================
# while True:
#     print("\n[f] Add Foreground ROI | [b] Remove Background ROI | [q] Result")
#     key = cv2.waitKey(0) & 0xFF

#     if key == ord('q'):
#         cv2.destroyAllWindows()
#         break

#     if key == ord('f'):
#         roi = select_roi_or_skip("FG ROI", img, "Select FG ROI")
#         if roi is None:
#             continue
#         x, y, w_, h_ = roi
#         mask[y:y+h_, x:x+w_] = cv2.GC_PR_FGD
#         cv2.grabCut(img, mask, None, bgdModel, fgdModel, ITER_GLOBAL, cv2.GC_INIT_WITH_MASK)
#         show_current(img, mask, "After FG ROI")

#     if key == ord('b'):
#         roi = select_roi_or_skip("BG ROI", img, "Select BG ROI")
#         if roi is None:
#             continue
#         x, y, w_, h_ = roi
#         mask[y:y+h_, x:x+w_] = cv2.GC_BGD
#         cv2.grabCut(img, mask, None, bgdModel, fgdModel, ITER_GLOBAL, cv2.GC_INIT_WITH_MASK)
#         show_current(img, mask, "After BG ROI")

# # ===============================
# # 5. Final result (Extraction)
# # ===============================
# final_mask = np.where(
#     (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
#     255, 0
# ).astype(np.uint8)

# extraction = cv2.bitwise_and(img, img, mask=final_mask)

# cv2.namedWindow("Extraction Result", cv2.WINDOW_NORMAL)
# cv2.imshow("Extraction Result", extraction)
# resize_window_keep_aspect("Extraction Result", extraction)
# cv2.waitKey(0)
# cv2.destroyWindow("Extraction Result")

# # ===============================
# # 6. ROI-driven Composition  ⭐重點改這裡
# # ===============================
# bg = cv2.imread("background.jpg")
# if bg is None:
#     raise SystemExit("Cannot find background.jpg")

# # ---- select ROI on background ----
# bg_roi = select_roi_or_skip(
#     "Select BG Placement ROI",
#     bg,
#     "Select ROI on background for placing target"
# )
# if bg_roi is None:
#     raise SystemExit("No background ROI selected")

# rx, ry, rw, rh = bg_roi

# # ---- scale foreground to fit ROI ----
# fg_h, fg_w = img.shape[:2]
# scale = min(rw / fg_w, rh / fg_h)

# new_w = int(fg_w * scale)
# new_h = int(fg_h * scale)

# fg_resized = cv2.resize(img, (new_w, new_h))
# mask_resized = cv2.resize(
#     final_mask,
#     (new_w, new_h),
#     interpolation=cv2.INTER_NEAREST
# )

# # ---- align center inside ROI ----
# x = rx + (rw - new_w) // 2
# y = ry + (rh - new_h) // 2

# # ---- composite ----
# canvas = bg.copy()
# roi_bg = canvas[y:y+new_h, x:x+new_w]

# mask_3c = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)

# fg_part = cv2.bitwise_and(fg_resized, mask_3c)
# bg_part = cv2.bitwise_and(roi_bg, cv2.bitwise_not(mask_3c))

# canvas[y:y+new_h, x:x+new_w] = cv2.add(fg_part, bg_part)

# # ===============================
# # 7. Display result
# # ===============================
# cv2.namedWindow("Composite Result", cv2.WINDOW_NORMAL)
# cv2.imshow("Composite Result", canvas)
# resize_window_keep_aspect("Composite Result", canvas)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
import cv2
import numpy as np

# 1. 讀取圖片
img = cv2.imread('gemini.png')
mask = np.zeros(img.shape[:2], np.uint8)

# 2. 定義背景與前景模型 (GrabCut 內部使用)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# 3. 設定感興趣的矩形區域 (Rect)
# 格式為 (x, y, width, height)。這裡我們縮減 10 像素邊界，確保主體在內
rect = (10, 10, img.shape[1]-20, img.shape[0]-20)

# 4. 執行 GrabCut
# 迭代 5 次，模式設為 cv2.GC_INIT_WITH_RECT
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

# 5. 修改遮罩
# GrabCut 會將像素分類為：
# 0: 確定背景, 1: 確定前景, 2: 可能背景, 3: 可能前景
# 我們將 0 和 2 設為 0 (背景)，1 和 3 設為 1 (前景)
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# 6. 將遮罩應用到原圖
result = img * mask2[:, :, np.newaxis]

# 7. 處理透明背景 (選用)
# 如果想要透明背景，可以增加 Alpha 通道
b, g, r = cv2.split(result)
alpha = mask2 * 255
final_rgba = cv2.merge([b, g, r, alpha])

# 儲存結果
cv2.imwrite('result.png', final_rgba)
print("處理完成，結果已儲存為 result.png")