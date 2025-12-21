import cv2
import numpy as np

# ===============================
# Config
# ===============================
ITER_INIT = 5
ITER_LOCAL = 5
ITER_GLOBAL = 5
DISPLAY_SIZE = (600, 800)

# ===============================
# Helper functions
# ===============================
def select_roi_or_skip(win, img, msg):
    print(msg)
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    rect = cv2.selectROI(win, img, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(win)

    x, y, w, h = rect
    if w <= 0 or h <= 0:
        return None

    H, W = img.shape[:2]
    x = max(0, x); y = max(0, y)
    w = min(w, W - x); h = min(h, H - y)
    if w <= 0 or h <= 0:
        return None

    return (x, y, w, h)

def show_current(img, mask, title="Current Result"):
    fg = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255, 0
    ).astype(np.uint8)

    result = cv2.bitwise_and(img, img, mask=fg)

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, result)
    cv2.resizeWindow(title, DISPLAY_SIZE[0], DISPLAY_SIZE[1])
    cv2.waitKey(1)

# ===============================
# 1. Read image
# ===============================
img = cv2.imread("target.jpg")
if img is None:
    raise SystemExit("找不到 target.jpg")

h, w = img.shape[:2]

# ===============================
# 2. Select PERSON ROI (mandatory)
# ===============================
person_roi = select_roi_or_skip(
    "Select PERSON ROI",
    img,
    "Step 1：請框選『人物』ROI（必選）"
)
if person_roi is None:
    raise SystemExit("未選人物 ROI，結束")

px, py, pw, ph = person_roi

# ===============================
# 3. Initial GrabCut (global)
# ===============================
mask = np.zeros((h, w), np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

cv2.grabCut(
    img, mask, (px, py, pw, ph),
    bgdModel, fgdModel, ITER_INIT,
    cv2.GC_INIT_WITH_RECT
)

show_current(img, mask, "After Initial GrabCut")

# ===============================
# 4. Interactive refinement loop
# ===============================
while True:
    print("\n========== 修正選單 ==========")
    print("[f] 補『前景』ROI（手臂、肩膀）")
    print("[b] 移除『背景』ROI（地板、泥土）")
    print("[q] 完成並輸出結果")
    print("==============================")

    key = cv2.waitKey(0) & 0xFF

    # ===========================
    # Quit
    # ===========================
    if key == ord('q'):
        break

    # ===========================
    # Foreground ROI (LOCAL GrabCut)
    # ===========================
    if key == ord('f'):
        fg_roi = select_roi_or_skip(
            "Select FG ROI",
            img,
            "框選『要補的前景 ROI』（Cancel 跳過）"
        )
        if fg_roi is None:
            continue

        cv2.destroyAllWindows()
        cv2.waitKey(1)
        
        fx, fy, fw, fh = fg_roi
        roi_img = img[fy:fy+fh, fx:fx+fw]
        rh, rw = roi_img.shape[:2]

        roi_mask = np.zeros((rh, rw), np.uint8)
        roi_bgd = np.zeros((1, 65), np.float64)
        roi_fgd = np.zeros((1, 65), np.float64)

        # 局部 GrabCut（假設 ROI 內有前景）
        cv2.grabCut(
            roi_img,
            roi_mask,
            (1, 1, rw-2, rh-2),
            roi_bgd, roi_fgd,
            ITER_LOCAL,
            cv2.GC_INIT_WITH_RECT
        )

        roi_fg = np.where(
            (roi_mask == cv2.GC_FGD) | (roi_mask == cv2.GC_PR_FGD),
            255, 0
        ).astype(np.uint8)

        # 合併回全域前景
        global_fg = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
            255, 0
        ).astype(np.uint8)

        global_fg[fy:fy+fh, fx:fx+fw] = cv2.bitwise_or(
            global_fg[fy:fy+fh, fx:fx+fw],
            roi_fg
        )

        mask[global_fg > 0] = cv2.GC_PR_FGD

        cv2.grabCut(
            img, mask, None,
            bgdModel, fgdModel,
            ITER_GLOBAL,
            cv2.GC_INIT_WITH_MASK
        )

        show_current(img, mask, "After FG ROI Refinement")

    # ===========================
    # Background ROI (LOCAL GrabCut)
    # ===========================
    if key == ord('b'):
        bg_roi = select_roi_or_skip(
            "Select BG ROI",
            img,
            "框選『要移除的背景 ROI』（Cancel 跳過）"
        )
        if bg_roi is None:
            continue
        
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        bx, by, bw, bh = bg_roi
        roi_img = img[by:by+bh, bx:bx+bw]
        rh, rw = roi_img.shape[:2]

        roi_mask = np.zeros((rh, rw), np.uint8)
        roi_bgd = np.zeros((1, 65), np.float64)
        roi_fgd = np.zeros((1, 65), np.float64)

        # 局部 GrabCut（假設 ROI 內是背景）
        cv2.grabCut(
            roi_img,
            roi_mask,
            (1, 1, rw-2, rh-2),
            roi_bgd, roi_fgd,
            ITER_LOCAL,
            cv2.GC_INIT_WITH_RECT
        )

        roi_bg = np.where(
            (roi_mask == cv2.GC_BGD) | (roi_mask == cv2.GC_PR_BGD),
            255, 0
        ).astype(np.uint8)

        # 從全域前景中扣掉背景
        global_fg = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
            255, 0
        ).astype(np.uint8)

        global_fg[by:by+bh, bx:bx+bw][roi_bg > 0] = 0

        # 更新 mask
        mask[:] = cv2.GC_BGD
        mask[global_fg > 0] = cv2.GC_PR_FGD

        cv2.grabCut(
            img, mask, None,
            bgdModel, fgdModel,
            ITER_GLOBAL,
            cv2.GC_INIT_WITH_MASK
        )

        show_current(img, mask, "After BG ROI Removal")

# ===============================
# 5. Final result
# ===============================
final_mask = np.where(
    (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
    255, 0
).astype(np.uint8)

result = cv2.bitwise_and(img, img, mask=final_mask)

cv2.namedWindow("Final Result", cv2.WINDOW_NORMAL)
cv2.imshow("Final Result", result)
cv2.resizeWindow("Final Result", DISPLAY_SIZE[0], DISPLAY_SIZE[1])
cv2.waitKey(0)
cv2.destroyAllWindows()
