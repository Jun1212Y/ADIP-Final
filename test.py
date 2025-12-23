import cv2
import numpy as np
import math

# =========================
# 0. PRE-PROCESS FUNCTION (必須放在 SETUP 前面)
# =========================
def preprocess_foreground(img):
    """結合裂痕修復與對比度增強"""
    if img is None: return None
    # 1. 裂痕修復 (Inpaint)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, crack_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    crack_mask = cv2.dilate(crack_mask, np.ones((3,3), np.uint8), iterations=1)
    img_fixed = cv2.inpaint(img, crack_mask, 3, cv2.INPAINT_TELEA)

    # 2. 對比度增強 (CLAHE)
    lab = cv2.cvtColor(img_fixed, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_enhanced = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(img_enhanced, cv2.COLOR_LAB2BGR)

# =========================
# 1. SETUP
# =========================
try:
    fg_img = cv2.imread("old_photo_dataset/12/0010053x1.png")
    bg_img = cv2.imread("old_photo_dataset/12/10080.jpg")

    # fg_img = cv2.imread("Dataset/adip_final_dataset/target.jpg")
    # bg_img = cv2.imread("Dataset/adip_final_dataset/background.jpg")

    # fg_img = cv2.imread("Dataset/adip_final_dataset/10116.jpg")
    # bg_img = cv2.imread("Dataset/adip_final_dataset/10117.jpg")
    
    # fg_img = cv2.imread("target.jpg")
    # bg_img = cv2.imread("background.jpg")
    if fg_img is not None:
    # 彈出視窗讓使用者決定
        cv2.namedWindow("Preprocess or not",cv2.WINDOW_NORMAL)
        cv2.imshow("Preprocess or not", fg_img)
        print(">>> 讀取成功。若需修復舊照裂痕/增強對比請按 'P'，否則按任意鍵跳過...")
        k = cv2.waitKey(0)
        if k == ord('p') or k == ord('P'):
            fg_img = preprocess_foreground(fg_img) # 直接沿用變數名
            print(">>> 預處理完成。")
        cv2.destroyAllWindows()
    if fg_img is None:
        fg_img = np.zeros((400, 400, 3), np.uint8)
        cv2.circle(fg_img, (200, 200), 100, (0, 0, 255), -1)
    if bg_img is None:
        bg_img = np.zeros((600, 800, 3), np.uint8)
        bg_img[:] = (50, 50, 50)
except Exception as e:
    print(f"Error loading images: {e}")
    exit()

# =========================
# 2. STATE MANAGEMENT
# =========================
state = {
    # --- GRABCUT / FOREGROUND STATE ---
    "fg_cut": None,         
    "fg_mask": None,        
    "fg_cut_mask": None, 
    "remove_fg_mode": False, 
    "extraction_mask" : None,  
    "drawing": False,
    "rect": (0,0,0,0),
    "ix": -1, "iy": -1,
    "initialized": False,
    "bgdModel": np.zeros((1, 65), np.float64),
    "fgdModel": np.zeros((1, 65), np.float64),
    "history": [],
    
    # --- PASTE STATE ---
    "paste_pos": (bg_img.shape[1]//2, bg_img.shape[0]//2), 
    "paste_scale": 0.5,
    "rotation": 0,          
    "flip_h": False, 
    
    # --- INTERACTION STATE ---
    "current_bbox": None,    
    "is_dragging": False, 
    "drag_offset": (0,0),    
    "is_resizing": False,    
    "is_rotating": False,    
    "resize_handle": None,   
    "handle_coords": {},
    "rot_handle_pos": (0,0), 
    "handle_size": 8,
    "active_box": None,      

    # --- PATCH TOOL STATE ---
    "remove_mode": False,       
    "patch_step": 0,            
    "target_rect": None,        
    "source_rect": None,         

    # Color harmonization
    "harmonize": True,

    # Adjust lighting
    "relight": True,      # Toggle on/off
    "light_angle": 45,     # Angle in degrees (45 = Top-Right Light)

    # Occlusion handling
    "occlusion_mask": np.zeros(bg_img.shape[:2], np.uint8), # Black canvas
    "edit_occlusion": False,  # Toggle painting mode
    "brush_size": 25,          # Size of the brush
    "erasing_occ": False,
    "mouse_pos": (0, 0)
}

# =========================
# 3. HELPER FUNCTIONS
# =========================
def save_state():
    """Saves the current mask state to history for UNDO."""
    if state["fg_mask"] is not None:
        state["history"].append({
            "mask": state["fg_mask"].copy(),
            "bgd": state["bgdModel"].copy(),
            "fgd": state["fgdModel"].copy(),
            "init": state["initialized"]
        })
        if len(state["history"]) > 10: state["history"].pop(0)

def restore_state():
    """Restores the previous state (UNDO)."""
    if len(state["history"]) > 0:
        prev = state["history"].pop()
        state["fg_mask"] = prev["mask"]
        state["bgdModel"] = prev["bgd"]
        state["fgdModel"] = prev["fgd"]
        state["initialized"] = prev["init"]
        update_cutout()
        print(">>> UNDO performed. Reverted to previous step.", flush=True)
    else:
        print(">>> Nothing to Undo.", flush=True)

def lock_foreground():
    """Confirms the current selection so future GrabCut steps won't change it."""
    if state["fg_mask"] is not None:
        save_state() # Save before locking in case we want to undo the lock
        
        # Convert "Probable" (GC_PR_FGD) to "Definite" (GC_FGD)
        # This tells the algorithm: "I am 100% sure this is foreground, don't touch it."
        state["fg_mask"] = np.where(state["fg_mask"] == cv2.GC_PR_FGD, cv2.GC_FGD, state["fg_mask"])
        state["fg_mask"] = np.where(state["fg_mask"] == cv2.GC_PR_BGD, cv2.GC_BGD, state["fg_mask"])
        
        print(">>> SELECTION LOCKED! You can now select the next body part safely.", flush=True)
        update_cutout()

def update_cutout():
    if state["fg_mask"] is None: return
    # We include both Definite (1) and Probable (3) foregrounds
    binary = np.where((state["fg_mask"] == cv2.GC_FGD) | (state["fg_mask"] == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    state["extraction_mask"] = (binary * 255).astype(np.uint8)
    ys, xs = np.where(binary == 1)
    if len(ys) > 0:
        y1, y2 = ys.min(), ys.max() + 1
        x1, x2 = xs.min(), xs.max() + 1
        state["fg_cut"] = fg_img[y1:y2, x1:x2].copy()
        state["fg_cut_mask"] = binary[y1:y2, x1:x2].copy()
        if state["fg_cut"].shape[0] * state["paste_scale"] < 10: state["paste_scale"] = 0.5
    else:
        state["fg_cut"] = None

def point_in_rect(point, rect):
    x, y = point
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh

def get_handles(x, y, w, h, size):
    return {
        'tl': (x, y),          'tr': (x + w, y),
        'bl': (x, y + h),      'br': (x + w, y + h)
    }

def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_LINEAR)

def match_color_stats(source, target):
    """
    Transfers the color atmosphere (mean & std dev) from the 'target' image 
    to the 'source' image.
    source: The foreground object (e.g., the person)
    target: The background region where the object will be pasted
    """
    # 1. Convert both images to LAB color space (L=Lightness, A/B=Color Channels)
    # LAB works better than RGB for color transfer because it separates light from color.
    src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

    # 2. Compute statistics for each channel
    src_mean, src_std = cv2.meanStdDev(src_lab)
    tgt_mean, tgt_std = cv2.meanStdDev(tgt_lab)

    # Flatten arrays for easier math
    src_mean, src_std = src_mean.flatten(), src_std.flatten()
    tgt_mean, tgt_std = tgt_mean.flatten(), tgt_std.flatten()

    # 3. Apply the statistical transfer equation:
    # New_Pixel = ( (Old_Pixel - Old_Mean) / Old_Std ) * New_Std + New_Mean
    res_lab = src_lab.copy()
    for i in range(3):
        # Avoid division by zero
        if src_std[i] == 0: src_std[i] = 1 
        
        # Normalize source to mean=0, std=1
        t = (src_lab[:, :, i] - src_mean[i]) / src_std[i]
        
        # Scale to target std and shift to target mean
        res_lab[:, :, i] = t * tgt_std[i] + tgt_mean[i]

    # 4. Clip values to valid 0-255 range and convert back to BGR
    res_lab = np.clip(res_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(res_lab, cv2.COLOR_LAB2BGR)

def apply_light_source(image, angle_deg, intensity=0.2): #0.4
    """
    Simulates directional light by applying a brightness gradient.
    angle_deg: Direction (0=Right, 90=Down, 180=Left, 270=Up)
    intensity: How strong the shadow/highlight contrast is (0.0 to 1.0)
    """
    h, w = image.shape[:2]
    if h == 0 or w == 0: return image
    
    # 1. Create a coordinate grid (-1 to 1)
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    X, Y = np.meshgrid(x, y)
    
    # 2. Calculate projection along the light angle
    # This creates a ramp: Positive values toward light, negative away
    rad = math.radians(angle_deg)
    projection = X * math.cos(rad) + Y * math.sin(rad)
    
    # 3. Create Gain Map (1.0 = Neutral, >1.0 = Bright, <1.0 = Dark)
    # We assume 'intensity' controls how much we brighten/darken
    gain_map = 1.0 + (projection * intensity)
    
    # 4. Apply to Value Channel (HSV)
    # We convert to HSV so we only change Brightness (V), not Color (H/S)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] *= gain_map # Multiply brightness
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

# =========================
# 4. MOUSE CALLBACKS
# =========================
def on_fg_click(event, x, y, flags, param):
    global state
    if event == cv2.EVENT_LBUTTONDOWN:
        state["drawing"] = True
        state["ix"], state["iy"] = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if state["drawing"]:
            state["rect"] = (state["ix"], state["iy"], x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        state["drawing"] = False
        w, h = abs(x - state["ix"]), abs(y - state["iy"])
        final_rect = (min(state["ix"], x), min(state["iy"], y), w, h)

        if w > 10 and h > 10:

        # =================================================
        # REMOVE WRONG FOREGROUND (SAFE VERSION)
        # =================================================
            if state.get("remove_fg_mode", False):
            # 🚧 保護：尚未初始化 GrabCut 時，不能移除
                if not state.get("initialized", False) or state.get("fg_mask") is None:
                    print(">>> Remove FG ignored: no foreground initialized yet.", flush=True)
                    return

                save_state()

                rx, ry, rw, rh = final_rect
                mask_h, mask_w = state["fg_mask"].shape
                ry2 = min(ry + rh, mask_h)
                rx2 = min(rx + rw, mask_w)

                state["fg_mask"][ry:ry2, rx:rx2] = cv2.GC_BGD

                cv2.grabCut(
                    fg_img,
                    state["fg_mask"],
                    None,
                    state["bgdModel"],
                    state["fgdModel"],
                    10,
                    cv2.GC_INIT_WITH_MASK
                )

                print(">>> Foreground region REMOVED.", flush=True)
                update_cutout()
                return
        # =================================================

        # ---------- Origin add foreground ----------
        save_state()

        if not state["initialized"]:
            state["fg_mask"] = np.zeros(fg_img.shape[:2], np.uint8)
            cv2.grabCut(
                fg_img,
                state["fg_mask"],
                final_rect,
                state["bgdModel"],
                state["fgdModel"],
                10,
                cv2.GC_INIT_WITH_RECT
            )
            state["initialized"] = True
            print(">>> Initial Cut Done.", flush=True)
        else:
            rx, ry, rw, rh = final_rect
            mask_h, mask_w = state["fg_mask"].shape
            ry2 = min(ry + rh, mask_h)
            rx2 = min(rx + rw, mask_w)
            roi = state["fg_mask"][ry:ry2, rx:rx2]

            state["fg_mask"][ry:ry2, rx:rx2] = np.where(
                roi == cv2.GC_FGD,
                cv2.GC_FGD,
                cv2.GC_PR_FGD
            )

            cv2.grabCut(
                fg_img,
                state["fg_mask"],
                None,
                state["bgdModel"],
                state["fgdModel"],
                5,
                cv2.GC_INIT_WITH_MASK
            )

            print(">>> Foreground updated.", flush=True)

        update_cutout()

def on_bg_click(event, x, y, flags, param):
    global state
    hs = state["handle_size"]
    interaction_consumed = False 

    state["mouse_pos"] = (x, y)

    # --- MOUSE WHEEL FOR BRUSH SIZE ---
    if state["edit_occlusion"] and event == cv2.EVENT_MOUSEWHEEL:
        # flags > 0 means Scroll Up (Grow), flags < 0 means Scroll Down (Shrink)
        if flags > 0:
            state["brush_size"] = min(state["brush_size"] + 5, 200)
        else:
            state["brush_size"] = max(state["brush_size"] - 5, 5)

    # --- OCCLUSION MASK EDITING ---
    if state["edit_occlusion"]:
        if event == cv2.EVENT_LBUTTONDOWN or (event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON):
            # Left Click: Paint barrier (Hide object)
            cv2.circle(state["occlusion_mask"], (x,y), state["brush_size"], 255, -1)
        elif event == cv2.EVENT_RBUTTONDOWN or (event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_RBUTTON):
            # Right Click: Erase barrier (Show object)
            cv2.circle(state["occlusion_mask"], (x,y), state["brush_size"], 0, -1)
        return # Stop here so we don't drag the object while painting

    # --- 1. PATCH TOOL INTERACTION ---
    if state["remove_mode"]:
        if state["patch_step"] == 0 or state["patch_step"] == 1:
            if event == cv2.EVENT_LBUTTONDOWN:
                state["drawing"] = True
                state["patch_step"] = 1
                state["ix"], state["iy"] = x, y
                interaction_consumed = True
            elif event == cv2.EVENT_MOUSEMOVE:
                if state["drawing"]: state["target_rect"] = (state["ix"], state["iy"], x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                state["drawing"] = False
                ix, iy = state["ix"], state["iy"]
                x1, y1, x2, y2 = min(ix, x), min(iy, y), max(ix, x), max(iy, y)
                if x2-x1 > 10 and y2-y1 > 10:
                    state["target_rect"] = (x1, y1, x2-x1, y2-y1)
                    state["source_rect"] = (x1 + (x2-x1) + 10, y1, x2-x1, y2-y1)
                    state["patch_step"] = 2 
        elif state["patch_step"] == 2:
            tx, ty, tw, th = state["target_rect"]
            sx, sy, sw, sh = state["source_rect"]
            boxes = {'source': (sx, sy, sw, sh), 'target': (tx, ty, tw, th)}

            if event == cv2.EVENT_LBUTTONDOWN:
                clicked_handle = False
                for b_name, b_rect in boxes.items():
                    bx, by, bw, bh = b_rect
                    handles = get_handles(bx, by, bw, bh, hs)
                    for h_name, (hx, hy) in handles.items():
                        if point_in_rect((x,y), (hx-hs, hy-hs, hs*2, hs*2)):
                            state["is_resizing"] = True
                            state["resize_handle"] = h_name
                            state["active_box"] = b_name 
                            state["ix"], state["iy"] = x, y
                            clicked_handle = True
                            interaction_consumed = True
                            break
                    if clicked_handle: break
                if not clicked_handle:
                    for b_name, b_rect in boxes.items():
                        if point_in_rect((x,y), b_rect):
                            state["is_dragging"] = True
                            state["active_box"] = b_name
                            state["drag_offset"] = (b_rect[0] - x, b_rect[1] - y)
                            interaction_consumed = True
                            break
            elif event == cv2.EVENT_MOUSEMOVE:
                if state["is_dragging"] and state["active_box"] in ['source', 'target']:
                    ox, oy = state["drag_offset"]
                    if state["active_box"] == 'source': state["source_rect"] = (x+ox, y+oy, sw, sh)
                    elif state["active_box"] == 'target': state["target_rect"] = (x+ox, y+oy, tw, th)
                elif state["is_resizing"] and state["active_box"] in ['source', 'target']:
                    dx, dy = x - state["ix"], y - state["iy"]
                    cx, cy, cw, ch = (sx, sy, sw, sh) if state["active_box"] == 'source' else (tx, ty, tw, th)
                    ncx, ncy, ncw, nch = cx, cy, cw, ch
                    if 'br' in state["resize_handle"]: ncw += dx; nch += dy
                    elif 'bl' in state["resize_handle"]: ncx += dx; ncw -= dx; nch += dy
                    elif 'tr' in state["resize_handle"]: ncy += dy; ncw += dx; nch -= dy
                    elif 'tl' in state["resize_handle"]: ncx += dx; ncy += dy; ncw -= dx; nch -= dy
                    if ncw > 10 and nch > 10:
                        new_rect = (ncx, ncy, ncw, nch)
                        if state["active_box"] == 'source': state["source_rect"], state["target_rect"] = new_rect, (tx, ty, ncw, nch)
                        else: state["target_rect"], state["source_rect"] = new_rect, (sx, sy, ncw, nch)
                        state["ix"], state["iy"] = x, y
            elif event == cv2.EVENT_LBUTTONUP:
                state["is_dragging"] = state["is_resizing"] = False
                state["active_box"] = None

    # --- 2. FOREGROUND OBJECT INTERACTION ---
    if not interaction_consumed and state["fg_cut"] is not None and state["current_bbox"] is not None:
        cx, cy = state["paste_pos"]
        bbox = state["current_bbox"] 
        rx, ry = state["rot_handle_pos"]

        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_something = False
            
            # A. CHECK ROTATION HANDLE
            if point_in_rect((x,y), (rx-hs, ry-hs, hs*2, hs*2)):
                state["is_rotating"] = True
                state["active_box"] = 'paste'
                clicked_something = True

            # B. CHECK RESIZE HANDLES
            if not clicked_something:
                for handle_name, (hx, hy) in state["handle_coords"].items():
                    if point_in_rect((x,y), (hx-hs, hy-hs, hs*2, hs*2)):
                        state["is_resizing"] = True
                        state["resize_handle"] = handle_name
                        state["active_box"] = 'paste'
                        clicked_something = True
                        break
            
            # C. CHECK DRAG BODY
            if not clicked_something:
                if point_in_rect((x,y), (bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1])):
                    state["is_dragging"] = True
                    state["active_box"] = 'paste'
                    state["drag_offset"] = (cx - x, cy - y)

        elif event == cv2.EVENT_LBUTTONUP:
            state["is_dragging"] = False
            state["is_resizing"] = False
            state["is_rotating"] = False
            state["active_box"] = None

        elif event == cv2.EVENT_MOUSEMOVE:
            if state["is_rotating"] and state["active_box"] == 'paste':
                delta_x, delta_y = x - cx, y - cy
                angle_deg = math.degrees(math.atan2(delta_y, delta_x)) + 90
                state["rotation"] = angle_deg
            elif state["is_dragging"] and state["active_box"] == 'paste':
                ox, oy = state["drag_offset"]
                state["paste_pos"] = (x + ox, y + oy)
            elif state["is_resizing"] and state["active_box"] == 'paste':
                curr_dist = math.sqrt((x - cx)**2 + (y - cy)**2)
                orig_h, orig_w = state["fg_cut"].shape[:2]
                base = math.sqrt((orig_w/2)**2 + (orig_h/2)**2)
                if base > 0: state["paste_scale"] = max(0.1, min(curr_dist/base, 5.0))

# =========================
# 5. MAIN LOOP
# =========================
cv2.namedWindow("Foreground", cv2.WINDOW_NORMAL)
cv2.namedWindow("Background", cv2.WINDOW_NORMAL)

if state["fg_cut"] is None and state["initialized"]: update_cutout()

cv2.setMouseCallback("Foreground", on_fg_click)
cv2.setMouseCallback("Background", on_bg_click)

print("--- CONTROLS ---")
print("[Left Drag] FG Window: Select parts of the object.")
print("[x]         REMOVE THE WRONG FOREGROUND TO BACKGROUND")
print("[SPACE]     LOCK SELECTION (Do this between steps!).")
print("[Z]         UNDO last mask change.")
print("[R]         Toggle Patch Mode.")
print("[ENTER]     In Patch Mode: Apply the patch.")
print("[H]         Toggle Color Harmonization.")
print("[S]         Save Result.")
print("----------------")

while True:
    # 1. Foreground Display
    display = fg_img.copy()
    if state["initialized"] and state["fg_mask"] is not None:
        display[state["fg_mask"] == cv2.GC_FGD] = [0, 255, 0]     
        display[state["fg_mask"] == cv2.GC_PR_FGD] = [255, 0, 255] 
    if state["drawing"] and not state["remove_mode"]:
        ix, iy, cx, cy = state["rect"]
        cv2.rectangle(display, (ix, iy), (cx, cy), (0, 255, 255), 2)
    
    cv2.putText(display, "Drag: Select  |  Space: Lock Part  |  Z: Undo", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow("Foreground", display)

    # 2. Background Composition
    final_comp = bg_img.copy()
    final_mask_layer = (
    state["extraction_mask"].copy()
    if state["extraction_mask"] is not None
    else np.zeros(fg_img.shape[:2], dtype=np.uint8)
    )

    hs = state["handle_size"]

    # ==========================================
    # 1. APPLY GLOBAL LIGHTING TO BACKGROUND
    # ==========================================
    # This changes the lighting of the world BEFORE the person enters.
    if state.get("relight", True):
        final_comp = apply_light_source(final_comp, state.get("light_angle", 45))
    # ==========================================

    # --- STEP A: BLEND THE PERSON ---
    if state["fg_cut"] is not None:
        cx, cy = state["paste_pos"]
        fg = state["fg_cut"]
        mask = state["fg_cut_mask"]
        
        h, w = fg.shape[:2]
        new_w = int(round(w * state["paste_scale"]))
        new_h = int(round(h * state["paste_scale"]))
        
        if new_w > 5 and new_h > 5:
            # Resize
            interp = cv2.INTER_AREA if state["paste_scale"] < 1.0 else cv2.INTER_CUBIC
            fg_s = cv2.resize(fg, (new_w, new_h), interpolation=interp)
            mask_u = (mask * 255).astype(np.uint8)
            mask_s = cv2.resize(mask_u, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Flip
            if state.get("flip_h", False):
                fg_s = cv2.flip(fg_s, 1)
                mask_s = cv2.flip(mask_s, 1)

            # Rotate
            if state["rotation"] != 0:
                fg_s = rotate_bound(fg_s, state["rotation"])
                mask_s = rotate_bound(mask_s, state["rotation"])

            rh, rw = fg_s.shape[:2]
            tx, ty = int(cx - rw/2), int(cy - rh/2)
            y1, y2 = max(ty, 0), min(ty + rh, final_comp.shape[0])
            x1, x2 = max(tx, 0), min(tx + rw, final_comp.shape[1])
            state["current_bbox"] = (x1, y1, x2, y2)
            
            if y2 > y1 and x2 > x1:
                sy1, sx1 = y1 - ty, x1 - tx
                sy2, sx2 = sy1 + (y2 - y1), sx1 + (x2 - x1)
                
                fg_crop = fg_s[sy1:sy2, sx1:sx2]
                mask_crop = mask_s[sy1:sy2, sx1:sx2]
                
                # CRITICAL: We grab 'roi' from 'final_comp', which is ALREADY LIT.
                # This means Color Harmony will adapt to the new lighting automatically!
                roi = final_comp[y1:y2, x1:x2]

                # Color Harmonization
                try:
                    fg_harmonized_full = match_color_stats(fg_crop, roi)
                except Exception:
                    fg_harmonized_full = fg_crop

                strength = state.get("harmony_strength", 0.6)
                if state.get("harmonize", True):
                    fg_to_blend = cv2.addWeighted(fg_crop, (1 - strength), fg_harmonized_full, strength, 0)
                else:
                    fg_to_blend = fg_crop

                # ==========================================
                # 2. APPLY GLOBAL LIGHTING TO FOREGROUND
                # ==========================================
                # We apply the SAME angle to the person so shadows match the background.
                if state.get("relight", True):
                    fg_to_blend = apply_light_source(fg_to_blend, state.get("light_angle", 45))
                # ==========================================

                # Blend
                alpha = mask_crop.astype(float) / 255.0
                alpha = np.repeat(alpha[:, :, None], 3, axis=2)

                occ_crop = state["occlusion_mask"][y1:y2, x1:x2]
                
                # 2. Invert: White (255) means "Blocked/Hidden", so visibility is 0.
                visibility = 1.0 - (occ_crop.astype(float) / 255.0)
                visibility = np.repeat(visibility[:, :, None], 3, axis=2)

                # 3. Multiply: If visibility is 0, alpha becomes 0 (Transparent)
                alpha = alpha * visibility

                final_comp[y1:y2, x1:x2] = (fg_to_blend * alpha + roi * (1.0 - alpha)).astype(np.uint8)

                # visible_mask_part = cv2.bitwise_and(mask_crop, cv2.bitwise_not(occ_crop))
                # final_mask_layer[y1:y2, x1:x2] = visible_mask_part

    # --- STEP B: VISUALIZATION COPY ---
    vis_comp = final_comp.copy()

    # Occlusion Mask Visualization
    if state["edit_occlusion"]:
        # Draw the mask in RED so you can see it
        red_map = np.zeros_like(vis_comp)
        red_map[:,:,2] = 255
        mask_bool = state["occlusion_mask"] > 0
        # Simple blend for display only
        vis_comp[mask_bool] = (0.5 * vis_comp[mask_bool] + 0.5 * red_map[mask_bool]).astype(np.uint8)
        
        cv2.putText(vis_comp, "PAINT MODE: Left=Hide, Right=Show", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # Brush Size Indicator
        mx, my = state["mouse_pos"]
        cv2.circle(vis_comp, (mx, my), state["brush_size"], (255,255,255), 1)
        cv2.putText(vis_comp, f"Size: {state['brush_size']}", (mx + 20, my), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        cv2.circle(vis_comp, (50, 100), state["brush_size"], (255,255,255), 1)

    # --- TEXT INSTRUCTIONS ---
    # Simplified instructions since "L" now controls everything
    info = f"R:Patch|H:Color|L:Global Light|F:Flip|S:Save"
    cv2.putText(vis_comp, info, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(vis_comp, "< > : Rotate Light Angle", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # --- STEP C: DRAW UI ---
    if state["remove_mode"]:
        cv2.putText(vis_comp, "PATCH TOOL: Select & Enter", (20, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        if state["patch_step"] == 1 and state["drawing"]:
            if state["target_rect"] is not None:
                ix, iy, cx, cy = state["target_rect"]
                cv2.rectangle(vis_comp, (ix, iy), (cx, cy), (0,0,255), 2)
        if state["patch_step"] == 2 and state["source_rect"] is not None:
            tx, ty, tw, th = state["target_rect"]
            sx, sy, sw, sh = state["source_rect"]
            valid = (sx >= 0 and sy >= 0 and sx+sw < bg_img.shape[1] and sy+sh < bg_img.shape[0])
            col = (0, 255, 0) if valid else (0, 0, 255)
            cv2.rectangle(vis_comp, (tx, ty), (tx+tw, ty+th), (0,0,255), 2)
            cv2.rectangle(vis_comp, (sx, sy), (sx+sw, sy+sh), col, 2)
            cv2.line(vis_comp, (sx+sw//2, sy+sh//2), (tx+tw//2, ty+th//2), (255,255,0), 1)
            handles = get_handles(sx, sy, sw, sh, hs)
            for (hx, hy) in handles.values():
                cv2.rectangle(vis_comp, (hx-hs, hy-hs), (hx+hs, hy+hs), (0,255,255), -1)

    elif state["fg_cut"] is not None and state["current_bbox"] is not None:
        x1, y1, x2, y2 = state["current_bbox"]
        cv2.rectangle(vis_comp, (x1, y1), (x2, y2), (255,200,0), 2)
        hc = get_handles(x1, y1, x2-x1, y2-y1, hs)
        state["handle_coords"] = hc
        for (hx, hy) in hc.values():
            cv2.rectangle(vis_comp, (hx-hs, hy-hs), (hx+hs, hy+hs), (0,255,255), -1)
            
        rot_x, rot_y = int((x1 + x2) / 2), y1 - 30 
        state["rot_handle_pos"] = (rot_x, rot_y)
        cv2.line(vis_comp, (rot_x, y1), (rot_x, rot_y), (255,200,0), 2)
        cv2.circle(vis_comp, (rot_x, rot_y), 8, (255, 0, 0), -1)
        cv2.circle(vis_comp, (rot_x, rot_y), 8, (255, 255, 255), 2)

    cv2.imshow("Background", vis_comp)

    # Input Handling
    key = cv2.waitKey(10) & 0xFF
    if key == 27: break
    elif key == ord('z'): restore_state()       
    elif key == ord(' '): lock_foreground()
    elif key == ord('x'):
        state["remove_fg_mode"] = not state["remove_fg_mode"]
        print(f">>> Remove Foreground Mode: {state['remove_fg_mode']}", flush=True)
    elif key == ord('r'):
        state["remove_mode"] = not state["remove_mode"]
        state["patch_step"] = 0
        state["target_rect"] = None
        state["source_rect"] = None
    
    elif key == ord('h'):
        state["harmonize"] = not state.get("harmonize", True)
    elif key == ord(']'):
        state["harmony_strength"] = min(state.get("harmony_strength", 0.6) + 0.1, 1.0)
    elif key == ord('['):
        state["harmony_strength"] = max(state.get("harmony_strength", 0.6) - 0.1, 0.0)
    
    # NEW FLIP KEY
    elif key == ord('f'):
        state["flip_h"] = not state.get("flip_h", False)

    # GLOBAL LIGHTING KEYS
    elif key == ord('l'): 
        state["relight"] = not state.get("relight", True)
    elif key == ord(','): 
        state["light_angle"] = (state.get("light_angle", 45) - 15) % 360
    elif key == ord('.'): 
        state["light_angle"] = (state.get("light_angle", 45) + 15) % 360

    # OCCLUSION EDITING KEY
    elif key == ord('o'): 
        state["edit_occlusion"] = not state["edit_occlusion"]
        print(f"Occlusion Mode: {state['edit_occlusion']}")

    # BRUSH SIZE KEYS
    elif key == ord('-'): # Decrease size
        state["brush_size"] = max(5, state["brush_size"] - 5)
    elif key == ord('='): # Increase size (using '=' so you don't have to hold shift for '+')
        state["brush_size"] = min(200, state["brush_size"] + 5)

    elif key == ord('s'):
        cv2.imwrite("saved_result.jpg", final_comp)
        cv2.imwrite("saved_mask.jpg", final_mask_layer)
        print(">>> Saved 'saved_result.jpg' and 'saved_mask.jpg'!")
    elif key == 13: # Enter
        if state["remove_mode"] and state["patch_step"] == 2:
            tx, ty, tw, th = state["target_rect"]
            sx, sy, sw, sh = state["source_rect"]
            if sx >= 0 and sy >= 0 and sx+sw < bg_img.shape[1] and sy+sh < bg_img.shape[0]:
                src_patch = bg_img[sy:sy+sh, sx:sx+sw]
                mask = 255 * np.ones(src_patch.shape, src_patch.dtype)
                center = (int(tx + tw/2), int(ty + th/2))
                try:
                    bg_img[:] = cv2.seamlessClone(src_patch, bg_img, mask, center, cv2.NORMAL_CLONE)
                    print("Patch Applied!")
                    state["patch_step"] = 0
                except Exception as e:
                    print(f"Clone Error: {e}")

cv2.destroyAllWindows()
