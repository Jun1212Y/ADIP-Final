import sys
import os
import cv2
import numpy as np
import math
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QSlider, QCheckBox, 
                             QFileDialog, QGroupBox, QScrollArea, QRadioButton, QButtonGroup, QMessageBox)
from PyQt6.QtCore import Qt, QPoint, QRect, QTimer
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QCursor

# ==========================================
# 1. HELPER FUNCTIONS (MATH & CV)
# ==========================================
class ImageUtils:
    @staticmethod
    def match_color_stats(source, target):
        """Transfers color atmosphere from target to source (LAB space)."""
        if source is None or target is None or source.size == 0 or target.size == 0:
            return source
        
        src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
        tgt_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

        src_mean, src_std = cv2.meanStdDev(src_lab)
        tgt_mean, tgt_std = cv2.meanStdDev(tgt_lab)

        src_mean, src_std = src_mean.flatten(), src_std.flatten()
        tgt_mean, tgt_std = tgt_mean.flatten(), tgt_std.flatten()

        res_lab = src_lab.copy()
        for i in range(3):
            if src_std[i] == 0: src_std[i] = 1 
            t = (src_lab[:, :, i] - src_mean[i]) / src_std[i]
            res_lab[:, :, i] = t * tgt_std[i] + tgt_mean[i]

        res_lab = np.clip(res_lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(res_lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def apply_light_source(image, angle_deg, intensity=0.2): #0.2
        """Applies directional lighting gradient."""
        h, w = image.shape[:2]
        if h == 0 or w == 0: return image
        
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        X, Y = np.meshgrid(x, y)
        
        rad = math.radians(angle_deg)
        projection = X * math.cos(rad) + Y * math.sin(rad)
        gain_map = 1.0 + (projection * intensity)
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] *= gain_map
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    @staticmethod
    def rotate_image(image, angle):
        if angle == 0: return image
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

    @staticmethod
    def is_old_photo(img):
        if img is None: return False
        
        # Convert to HSV to analyze color
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        mean_s = np.mean(s)
        mean_h = np.mean(h)
        std_v = np.std(v) # Standard Deviation of brightness (Contrast)

        is_sepia = (10 <= mean_h <= 35) and (20 <= mean_s <= 90)
        is_faded_bw = (mean_s < 15) and (std_v < 60)
        
        return is_sepia or is_faded_bw
    
    @staticmethod
    def repair_old_photo(img):
        if img is None: return None
        
        # Repair Steps:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, crack_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        crack_mask = cv2.dilate(crack_mask, kernel, iterations=1)
        img_fixed = cv2.inpaint(img, crack_mask, 3, cv2.INPAINT_TELEA)
        
        # Enhance Contrast using CLAHE
        lab = cv2.cvtColor(img_fixed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        img_enhanced = cv2.merge((cl, a, b))
        return cv2.cvtColor(img_enhanced, cv2.COLOR_LAB2BGR)

# ==========================================
# 2. CANVAS WIDGET
# ==========================================
class CanvasWidget(QWidget):
    def __init__(self, main_window, canvas_type="bg"):
        super().__init__()
        self.main = main_window
        self.canvas_type = canvas_type # "fg" or "bg"
        self.setMouseTracking(True)
        
        self.display_pixmap = None
        self.img_scale = 1.0
        self.img_offset_x = 0
        self.img_offset_y = 0

        # Interaction State
        self.is_dragging = False
        self.drag_start = None
        self.current_rect = None # For drawing box visuals

        self.last_mouse_pos = None
        self.setMouseTracking(True) 
        self.drag_start = None
        self.current_rect = None 
        self.zoom_factor = 1.0  
        self.setMouseTracking(True)
        self.pan_offset = QPoint(0, 0)
        self.is_panning = False
        self.last_mouse_pan_pos = QPoint(0, 0)

    def update_display(self, cv_img):
        if cv_img is None: return
        h, w, ch = cv_img.shape
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        qt_img = QImage(rgb_img.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.display_pixmap = QPixmap.fromImage(qt_img)
        self.update()

    def get_img_coords(self, widget_pos):
        if not self.display_pixmap or self.img_scale == 0:
            return 0, 0
            
        # Convert widget coordinates to image coordinates
        ix = int((widget_pos.x() - self.img_offset_x) / self.img_scale)
        iy = int((widget_pos.y() - self.img_offset_y) / self.img_scale)
        
        # Clamp to image bounds
        ix = max(0, min(ix, self.display_pixmap.width() - 1))
        iy = max(0, min(iy, self.display_pixmap.height() - 1))
        return ix, iy
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(40, 40, 40))

        if self.display_pixmap:
            # --- 1. Draw Image with Zoom & Pan ---
            # calculate scaled pixmap based on zoom factor ---
            base_pixmap = self.display_pixmap.scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
            final_w = int(base_pixmap.width() * self.zoom_factor)
            final_h = int(base_pixmap.height() * self.zoom_factor)
            scaled_pixmap = base_pixmap.scaled(final_w, final_h, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)

            # 2. Calculate offsets for centering + panning
            self.img_offset_x = (self.width() - scaled_pixmap.width()) // 2 + self.pan_offset.x()
            self.img_offset_y = (self.height() - scaled_pixmap.height()) // 2 + self.pan_offset.y()
            
            # 3. Calculate image scale for coordinate mapping
            self.img_scale = scaled_pixmap.width() / max(1, self.display_pixmap.width())
            
            painter.drawPixmap(self.img_offset_x, self.img_offset_y, scaled_pixmap)

            # --- 2. Draw Brush Preview (FG WINDOW) ---
            if self.canvas_type == "fg" and self.main.current_mode == "normal" and self.main.chk_brush.isChecked():
                if self.last_mouse_pos is not None:
                    display_radius = int(self.main.brush_size * self.img_scale)
                    pen = QPen(QColor(0, 255, 255, 255), 2, Qt.PenStyle.SolidLine)
                    painter.setPen(pen)
                    painter.setBrush(QColor(0, 255, 255, 50)) 
                    painter.drawEllipse(self.last_mouse_pos, display_radius, display_radius)
                    
                    # Crosshair
                    painter.drawLine(self.last_mouse_pos.x() - 5, self.last_mouse_pos.y(),
                                    self.last_mouse_pos.x() + 5, self.last_mouse_pos.y())
                    painter.drawLine(self.last_mouse_pos.x(), self.last_mouse_pos.y() - 5,
                                    self.last_mouse_pos.x(), self.last_mouse_pos.y() + 5)

            # --- DRAW TRANSFORM HANDLES (RIGHT WINDOW) ---
            if self.canvas_type == "bg" and self.main.current_mode == "normal" and self.main.active_geom:
                cx, cy, w, h = self.main.active_geom
                sx = int(cx * self.img_scale + self.img_offset_x)
                sy = int(cy * self.img_scale + self.img_offset_y)
                sw = int(w * self.img_scale)
                sh = int(h * self.img_scale)
                
                left, top = sx - sw // 2, sy - sh // 2
                right, bottom = sx + sw // 2, sy + sh // 2

                pen = QPen(QColor(0, 255, 255), 2, Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(left, top, sw, sh)

                handle_size = 10
                painter.setBrush(QColor(255, 255, 255))
                painter.setPen(QColor(0, 0, 0))
                corners = [(left, top), (right, top), (left, bottom), (right, bottom)]
                for (hx, hy) in corners:
                    painter.drawRect(hx - handle_size//2, hy - handle_size//2, handle_size, handle_size)

                painter.setBrush(QColor(255, 100, 100))
                rot_x, rot_y = int(sx), int(top - 30)
                painter.drawLine(int(sx), int(top), rot_x, rot_y)
                painter.drawEllipse(QPoint(rot_x, rot_y), 6, 6)

            # 2. Draw Visuals (Generic Rectangles)
            if self.current_rect and not self.main.is_picking_harmony:
                pen = QPen(QColor(0, 255, 255), 2, Qt.PenStyle.SolidLine)
                painter.setPen(pen)
                rx = int(self.current_rect.x() * self.img_scale + self.img_offset_x)
                ry = int(self.current_rect.y() * self.img_scale + self.img_offset_y)
                rw = int(self.current_rect.width() * self.img_scale)
                rh = int(self.current_rect.height() * self.img_scale)
                painter.drawRect(rx, ry, rw, rh)

            # 3. Patch Tool Visuals
            if self.canvas_type == "bg" and self.main.current_mode == "patch":
                # ... (Keep existing patch visuals) ...
                if self.main.patch_target:
                    pen = QPen(QColor(0, 0, 255), 2, Qt.PenStyle.DashLine)
                    painter.setPen(pen)
                    r = self.main.patch_target
                    rx = int(r.x() * self.img_scale + self.img_offset_x)
                    ry = int(r.y() * self.img_scale + self.img_offset_y)
                    painter.drawRect(rx, ry, int(r.width()*self.img_scale), int(r.height()*self.img_scale))
                
                if self.main.patch_source:
                    pen = QPen(QColor(0, 255, 0), 2, Qt.PenStyle.DashLine)
                    painter.setPen(pen)
                    r = self.main.patch_source
                    rx = int(r.x() * self.img_scale + self.img_offset_x)
                    ry = int(r.y() * self.img_scale + self.img_offset_y)
                    painter.drawRect(rx, ry, int(r.width()*self.img_scale), int(r.height()*self.img_scale))

            # 4. Occlusion Brush
            if self.canvas_type == "bg" and self.main.current_mode == "occlusion":
                # ... (Keep existing occlusion brush) ...
                pen = QPen(QColor(255, 255, 255), 1)
                painter.setPen(pen)
                mx = self.mapFromGlobal(QCursor.pos()).x()
                my = self.mapFromGlobal(QCursor.pos()).y()
                painter.drawEllipse(QPoint(mx, my), self.main.brush_size, self.main.brush_size)

            if self.canvas_type == "bg":
                # 1. Draw the box while dragging (Yellow Dash)
                if self.main.is_picking_harmony and self.current_rect:
                    pen = QPen(QColor(255, 255, 0), 2, Qt.PenStyle.DashLine)
                    painter.setPen(pen)
                    rx = int(self.current_rect.x() * self.img_scale + self.img_offset_x)
                    ry = int(self.current_rect.y() * self.img_scale + self.img_offset_y)
                    painter.drawRect(rx, ry, int(self.current_rect.width()*self.img_scale), int(self.current_rect.height()*self.img_scale))
                
                # 2. Draw the finalized saved box (Solid Gold)
                elif (self.main.chk_custom_harmony.isChecked() and self.main.custom_harmony_rect and self.main.is_box_visible):
                    cx, cy, cw, ch = self.main.custom_harmony_rect
                    pen = QPen(QColor(255, 215, 0), 2, Qt.PenStyle.SolidLine)
                    painter.setPen(pen)
                    sx = int(cx * self.img_scale + self.img_offset_x)
                    sy = int(cy * self.img_scale + self.img_offset_y)
                    painter.drawRect(sx, sy, int(cw*self.img_scale), int(ch*self.img_scale))
                    painter.drawText(sx, sy - 5, "Color Source")

                elif self.main.current_mode == "patch":
                    for attr, color in [('patch_target', QColor(0,0,255)), ('patch_source', QColor(0,255,0))]:
                        rect = getattr(self.main, attr)
                        if rect:
                            painter.setPen(QPen(color, 2, Qt.PenStyle.DashLine))
                            painter.drawRect(int(rect.x()*self.img_scale + self.img_offset_x),
                                            int(rect.y()*self.img_scale + self.img_offset_y),
                                            int(rect.width()*self.img_scale), int(rect.height()*self.img_scale))
        else:
            painter.setPen(QColor(200, 200, 200))
            label = "FOREGROUND SOURCE" if self.canvas_type == "fg" else "BACKGROUND"
            painter.drawText(10, 20, label)
            return

        # Keep the global text labels at the end
        painter.setPen(QColor(200, 200, 200))
        label = "FOREGROUND SOURCE" if self.canvas_type == "fg" else "BACKGROUND"
        painter.drawText(10, 20, label)

    def mousePressEvent(self, event):
        if not self.display_pixmap: return
        ix, iy = self.get_img_coords(event.pos())
        mx, my = event.pos().x(), event.pos().y() # Screen coords for handles
        
        # --- FG WINDOW ---
        if self.canvas_type == "fg":
            self.is_dragging = True
            self.drag_start = (ix, iy)
            self.current_rect = QRect(ix, iy, 0, 0)

            # 1. If in Brush Mode (Checked)
            if self.main.chk_brush.isChecked():
                is_fg = (event.button() == Qt.MouseButton.LeftButton)
                self.main.paint_grabcut_brush(ix, iy, is_fg)

            # 2. If in ROI Mode (Unchecked)
            else: 
                if event.button() == Qt.MouseButton.LeftButton:
                    self.current_rect = QRect(ix, iy, 0, 0)

        # --- BG WINDOW ---
        elif self.canvas_type == "bg":

            if self.main.is_picking_harmony:
                self.is_dragging = True
                self.drag_start = (ix, iy)
                self.current_rect = QRect(ix, iy, 0, 0)
                return # Stop here, don't trigger move/resize

            # 1. Check Transform Handles (Standard Mode)
            if self.main.current_mode == "normal" and self.main.active_geom:
                cx, cy, w, h = self.main.active_geom
                
                # Calculate Screen Coords of handles
                sx = int(cx * self.img_scale + self.img_offset_x)
                sy = int(cy * self.img_scale + self.img_offset_y)
                sw, sh = int(w * self.img_scale), int(h * self.img_scale)
                left, top, right, bottom = sx - sw//2, sy - sh//2, sx + sw//2, sy + sh//2
                
                # HIT TEST: Rotation Handle
                rot_x, rot_y = int(sx), int(top - 30)
                if abs(mx - rot_x) < 15 and abs(my - rot_y) < 15:
                    self.main.interaction_mode = 'rotate'
                    self.main.drag_start_val = (mx, my) # Store start pos
                    return

                # HIT TEST: Resize Handles (Corners)
                corners = [(left, top), (right, top), (left, bottom), (right, bottom)]
                for (hx, hy) in corners:
                    if abs(mx - hx) < 15 and abs(my - hy) < 15:
                        self.main.interaction_mode = 'resize'
                        # Store starting distance from center
                        self.main.drag_start_dist = math.hypot(mx - sx, my - sy)
                        self.main.drag_start_scale = self.main.scale
                        return

            # 2. Check Standard Tools
            if self.main.current_mode == "occlusion":
                is_erasing = (event.button() == Qt.MouseButton.RightButton)
                self.main.paint_occlusion(ix, iy, is_erasing, is_new_stroke=True)
                self.is_dragging = True
            elif self.main.current_mode == "patch":
                self.is_dragging = True
                self.drag_start = (ix, iy)
                self.current_rect = QRect(ix, iy, 0, 0)
            else:
                # Default: Move Object
                if event.button() == Qt.MouseButton.LeftButton:
                    self.main.interaction_mode = 'move'
                    self.main.start_dragging_object(ix, iy)

        # MiddleButton of Mouse to reset photo Zoom factor 
        if event.button() == Qt.MouseButton.MiddleButton:
            # 1. Ctrl + Middle Click: Reset View
            if QApplication.keyboardModifiers() == Qt.KeyboardModifier.ControlModifier:
                self.zoom_factor = 1.0
                self.pan_offset = QPoint(0, 0)
                print(">>> View Reset")
                self.update()
                return 
            
            # 2. Otherwise: Start Panning
            self.is_panning = True
            self.last_mouse_pan_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

    def mouseMoveEvent(self, event):
        self.last_mouse_pos = event.pos() # For brush preview
        self.update() 

        if not self.display_pixmap: return
        ix, iy = self.get_img_coords(event.pos())
        mx, my = event.pos().x(), event.pos().y()

        # FG Logic
        if self.canvas_type == "fg" and self.is_dragging:
            # CaseA: Standard Mode
            if self.main.current_mode == "normal":
                
                # A-1: ROI selection mode (Brush is Unchecked)
                if not self.main.chk_brush.isChecked() and (event.buttons() & Qt.MouseButton.LeftButton):
                    sx, sy = self.drag_start
                    self.current_rect = QRect(QPoint(sx, sy), QPoint(ix, iy)).normalized()
                    self.update()

                # A-2: Brush mode (Brush is Checked)
                elif self.main.chk_brush.isChecked():
                    self.current_rect = None  
                    if event.buttons() & Qt.MouseButton.LeftButton:
                        self.main.paint_grabcut_brush(ix, iy, is_fg=True)
                    elif event.buttons() & Qt.MouseButton.RightButton:
                        self.main.paint_grabcut_brush(ix, iy, is_fg=False)

        # BG Logic
        elif self.canvas_type == "bg":
            # HANDLE TRANSFORM INTERACTIONS            
            if self.main.is_picking_harmony and self.is_dragging:
                sx, sy = self.drag_start
                self.current_rect = QRect(QPoint(sx, sy), QPoint(ix, iy)).normalized()
                self.update()
                return
            
            if hasattr(self.main, 'interaction_mode'):
            # A. ROTATE
                if self.main.interaction_mode == 'rotate':
                    # Calculate angle relative to object center
                    if self.main.active_geom:
                        cx, cy, _, _ = self.main.active_geom
                        sx = int(cx * self.img_scale + self.img_offset_x)
                        sy = int(cy * self.img_scale + self.img_offset_y)
                        
                        # Angle logic
                        angle = math.degrees(math.atan2(my - sy, mx - sx))
                        self.main.rotation = int(angle + 90) # Offset to make top = 0
                        self.main.process_composition()
                        return

                # B. RESIZE
                elif self.main.interaction_mode == 'resize':
                    if self.main.active_geom:
                        cx, cy, _, _ = self.main.active_geom
                        sx = int(cx * self.img_scale + self.img_offset_x)
                        sy = int(cy * self.img_scale + self.img_offset_y)
                        
                        current_dist = math.hypot(mx - sx, my - sy)
                        # Ratio of new distance vs old distance
                        ratio = current_dist / max(1, self.main.drag_start_dist)
                        
                        # Apply to original scale
                        new_scale = self.main.drag_start_scale * ratio
                        self.main.scale = max(0.05, min(5.0, new_scale))
                        self.main.process_composition()
                        return
                
                # C. MOVE
                elif self.main.interaction_mode == 'move':
                    self.main.drag_object(ix, iy)
                    return

                # Visual Updates for other tools
                if self.main.current_mode == "occlusion":
                    self.update()
                    if event.buttons() & Qt.MouseButton.LeftButton: self.main.paint_occlusion(ix, iy, False, is_new_stroke=False)
                    elif event.buttons() & Qt.MouseButton.RightButton: self.main.paint_occlusion(ix, iy, True, is_new_stroke=False)
                elif self.main.current_mode == "patch" and self.is_dragging:
                    sx, sy = self.drag_start
                    self.current_rect = QRect(QPoint(sx, sy), QPoint(ix, iy)).normalized()
                    self.update()
        #Panning Move
        self.last_mouse_pos = event.pos()
        if self.is_panning:
           # 計算本次移動的差距
            delta = event.pos() - self.last_mouse_pan_pos
            self.pan_offset += delta
            self.last_mouse_pan_pos = event.pos()
            self.update() # 觸發重繪

        if self.main.is_drawing:
            # 根據按住的是左鍵還是右鍵決定前/背景
            is_fg = bool(event.buttons() & Qt.MouseButton.LeftButton)
            # 持續在 Mask 上塗抹，但不執行耗時運算
            self.main.paint_grabcut_brush(ix, iy, is_fg=is_fg)
            self.update()

        return

    def mouseReleaseEvent(self, event):
        # Reset Interaction Modes
        if hasattr(self.main, 'interaction_mode'):
            self.main.interaction_mode = None
        
        # Reset Dragging Flags
        if self.is_dragging:
            self.is_dragging = False

            if self.canvas_type == "fg" and self.main.current_mode == "normal":
                # FG ROI Selection Complete
                if not self.main.chk_brush.isChecked() and event.button() == Qt.MouseButton.LeftButton:
                    if self.current_rect and self.current_rect.width() > 5:
                        self.main.perform_grabcut(self.current_rect)
                    self.current_rect = None

            if self.canvas_type == "bg" and self.main.is_picking_harmony and self.current_rect:
                # Save the rect coordinates
                x, y, w, h = self.current_rect.getRect()
                # Basic validation
                if w > 5 and h > 5:
                    self.main.custom_harmony_rect = (x, y, w, h)
                    #self.main.is_picking_harmony = False # Turn off picking mode automatically
                    print("Custom Color Region Set.")
                    self.main.process_composition()
                self.current_rect = None
                self.update()
                return
                
            # Apply Cut/Patch logic...
            if self.canvas_type == "fg" and self.current_rect:
                if self.current_rect.width() > 5: self.main.perform_grabcut(self.current_rect)
                self.current_rect = None; self.update()
            elif self.canvas_type == "bg" and self.main.current_mode == "patch" and self.current_rect:
                if self.current_rect.width() > 5: self.main.register_patch_rect(self.current_rect)
                self.current_rect = None; self.update()
            elif self.canvas_type == "bg" and self.main.current_mode == "normal":
                self.main.stop_dragging()
        if event.button() == Qt.MouseButton.MiddleButton:
            self.is_panning = False
            self.setCursor(Qt.CursorShape.CrossCursor) # 恢復十字準星
            return

    def wheelEvent(self, event):
        # Photo Window Zoom
        # 1. Check for Ctrl key
        modifiers = QApplication.keyboardModifiers()
        
        # Ctrl + Wheel = Zoom View
        if modifiers == Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            
            # Adjust zoom factor
            step = 1.1 if delta > 0 else 0.9
            self.zoom_factor = max(0.2, min(self.zoom_factor * step, 10.0))
            
            # Adjust pan offset to zoom towards cursor
            if 0.95 <= self.zoom_factor <= 1.05:
                self.zoom_factor = 1.0
                self.pan_offset = QPoint(0, 0)
            
            self.update()
            event.accept()
            
        # 2. Additional Brush Size Adjustments
        elif self.canvas_type == "fg" and self.main.current_mode == "normal" and self.main.chk_brush.isChecked():
            delta = event.angleDelta().y()
            if delta > 0:
                self.main.brush_size = min(150, self.main.brush_size + 2)
            else:
                self.main.brush_size = max(1, self.main.brush_size - 2)
            self.update()
            event.accept()

        # 3. Additional Brush Size Adjustments
        elif self.canvas_type == "fg" and self.main.current_mode == "normal" and self.main.chk_brush.isChecked():
            delta = event.angleDelta().y()
            if delta > 0:
                self.main.brush_size = min(150, self.main.brush_size + 2)
            else:
                self.main.brush_size = max(1, self.main.brush_size - 2)
            self.update()
            event.accept()
            
        #FG Brush mode Zoom
        if self.canvas_type == "fg" and self.main.current_mode == "normal" and self.main.chk_brush.isChecked():
            # Get the wheel delta
            delta = event.angleDelta().y()
            
            # Adjust brush size
            if delta > 0:
                self.main.brush_size = min(150, self.main.brush_size + 2)
            else:
                self.main.brush_size = max(1, self.main.brush_size - 2)
            
            # Update the brush size slider in the main window, if it exists
            if hasattr(self.main, 'sld_brush'):
                self.main.sld_brush.setValue(self.main.brush_size)
            
            # Redraw the canvas to reflect the new brush size
            self.update()
            
            # Consume the event
            event.accept()
        else:
            # Default behavior
            super().wheelEvent(event)

        if self.canvas_type == "bg":
            delta = event.angleDelta().y()
            if self.main.current_mode == "occlusion":
                change = 5 if delta > 0 else -5
                self.main.update_brush_size(change)
                self.update()
            elif self.main.current_mode == "move":
                change = 0.05 if delta > 0 else -0.05
                self.main.update_scale_scroll(change)

# ==========================================
# 3. MAIN APPLICATION WINDOW
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ADIP Final Project - Interactive Image Compositor")
        self.resize(1400, 900)

        # --- STATE ---
        self.fg_img_orig = None
        self.bg_img_orig = None
        self.custom_harmony_rect = None  # Stores the rectangle (x, y, w, h)
        self.is_picking_harmony = False  # Flag for the drawing mode
        self.is_box_visible = False
        
        # GrabCut State
        self.mask = None
        self.bgdModel = np.zeros((1, 65), np.float64)
        self.fgdModel = np.zeros((1, 65), np.float64)
        self.is_drawing = False
        self.last_roi = None
        self.brush_size = 20 
        self.gc_initialized = False
        self.history = []
        
        # Composite State
        self.fg_cutout = None
        self.fg_cut_mask = None
        self.pos_x, self.pos_y = 0, 0
        self.scale = 0.5
        self.rotation = 0
        self.flip_h = False
        self.current_mode = "normal" # cut, move, patch, occlusion

        # Patch State
        self.patch_step = 0 # 0=None, 1=Target Set, 2=Source Set
        self.patch_target = None
        self.patch_source = None

        # Occlusion State
        self.occlusion_mask = None
        self.brush_size = 25

        # Dragging
        self.is_dragging_obj = False
        self.drag_offset = (0,0)

        self.setup_ui()
        self.load_defaults()
    
    def paint_grabcut_brush(self, x, y, is_fg=True):
        if self.mask is None: return
    
    # 1. Make sure to get the correct GrabCut value
        val = cv2.GC_FGD if is_fg else cv2.GC_BGD
    
    # 2. Draw circle on mask
        ix, iy = int(x), int(y)
        cv2.circle(self.mask, (ix, iy), self.brush_size, val, -1, lineType=cv2.LINE_8)
    
    # 3. Update GrabCut mask display
    #   For debugging: print actual mask value at center
        actual_val = self.mask[iy, ix]
        print(f"Brush applied at ({ix}, {iy}), Target: {val}, Actual in mask: {actual_val}")
        self.update_cutout_from_mask()
    
    def run_grabcut_iteration(self):
        if not hasattr(self, 'last_roi') or self.last_roi is None or self.mask is None:
            print(">>> 請先框選 ROI 區域")
            return

        try:
            print(">>> 執行局部區域 (N鍵) 更新...")
            r = self.last_roi
            h_img, w_img = self.fg_img_orig.shape[:2]

            # calc coordinates
            x1 = max(0, int(r.x()))
            y1 = max(0, int(r.y()))
            x2 = min(w_img, int(r.x() + r.width()))
            y2 = min(h_img, int(r.y() + r.height()))

            # 1. Capture ROI 圖片和 Mask
            img_roi = self.fg_img_orig[y1:y2, x1:x2]
            mask_roi = self.mask[y1:y2, x1:x2]

            # 2. Implement GrabCut on ROI
            cv2.grabCut(img_roi, mask_roi, None, 
                        self.bgdModel, self.fgdModel, 5, cv2.GC_INIT_WITH_MASK)
            
            # 4. Update the main mask with modified ROI
            self.mask[y1:y2, x1:x2] = mask_roi
        except Exception as e:
            print(f"N-Key Update Error: {e}")

    def save_state(self):
        """Captures a snapshot of the entire current state."""
        state_snapshot = {
            # 1. Foreground Selection State
            "gc_mask": self.mask.copy() if self.mask is not None else None,
            "gc_bgd": self.bgdModel.copy(),
            "gc_fgd": self.fgdModel.copy(),
            "gc_init": self.gc_initialized,
            
            # 2. Background Image State (for Patch Tool changes)
            "bg_img": self.bg_img_orig.copy() if self.bg_img_orig is not None else None,
            
            # 3. Occlusion Mask State (for Paint changes)
            "occ_mask": self.occlusion_mask.copy() if self.occlusion_mask is not None else None,
            
            # 4. Transform State (so objects don't jump around on undo)
            "transform_pos": (self.pos_x, self.pos_y),
            "transform_scale": self.scale,
            "transform_rot": self.rotation,
            "transform_flip": self.flip_h
        }

        self.history.append(state_snapshot)
        
        # Limit history to 20 to prevent memory crashes
        if len(self.history) > 20: 
            self.history.pop(0)

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # Canvas Layout
        canvas_layout = QHBoxLayout()
        self.canvas_fg = CanvasWidget(self, "fg")
        self.canvas_bg = CanvasWidget(self, "bg")
        canvas_layout.addWidget(self.canvas_fg)
        canvas_layout.addWidget(self.canvas_bg)
        layout.addLayout(canvas_layout, stretch=4)

        # Sidebar
        sidebar = QScrollArea()
        sb_widget = QWidget()
        sb_layout = QVBoxLayout(sb_widget)
        sb_widget.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid gray; margin-top: 10px; }")

        # --- 1. LOADERS ---
        g1 = QGroupBox("1. Images")
        l1 = QVBoxLayout()

        btn_fg = QPushButton("Load Person (FG)")
        btn_fg.clicked.connect(self.load_fg)
        btn_fg.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        btn_bg = QPushButton("Load Background (BG)")
        btn_bg.clicked.connect(self.load_bg)
        btn_bg.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        l1.addWidget(btn_fg); l1.addWidget(btn_bg)
        g1.setLayout(l1)
        sb_layout.addWidget(g1)

        # --- 2. MODES & TOOLS ---
        g2 = QGroupBox("2. Modes")
        l2 = QVBoxLayout() 
        self.btn_grp = QButtonGroup()

        # ===============================================
        # MODE A: Standard Interaction
        # ===============================================
        self.rb_normal = QRadioButton("Standard Interaction")
        self.rb_normal.setChecked(True)
        self.rb_normal.toggled.connect(lambda: self.set_mode("normal"))
        self.rb_normal.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        l2.addWidget(self.rb_normal)

        # Container for Standard Controls (Nested immediately under the Radio Button)
        self.container_normal = QWidget()
        lay_normal = QVBoxLayout(self.container_normal)
        lay_normal.setContentsMargins(20, 0, 0, 10) # Indent left by 20px

        # 2. Refine Brush Checkbox
        self.chk_brush = QRadioButton("Use Refine Brush")
        self.chk_brush.toggled.connect(lambda: self.canvas_fg.update()) 
        self.chk_brush.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        lay_normal.addWidget(self.chk_brush)

        # 3. Lock Button
        btn_lock = QPushButton("Lock / Freeze Selection (SPACE)")
        btn_lock.clicked.connect(self.lock_selection)
        btn_lock.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        lay_normal.addWidget(btn_lock)

        btn_undo_std = QPushButton("Undo Last Cut (Z)")
        btn_undo_std.clicked.connect(self.undo_cut)
        btn_undo_std.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        lay_normal.addWidget(btn_undo_std)
        
        l2.addWidget(self.container_normal)

        # ===============================================
        # MODE B: Patch Tool
        # ===============================================
        self.rb_patch = QRadioButton("Patch Tool (R)")
        self.rb_patch.toggled.connect(lambda: self.set_mode("patch"))
        self.rb_patch.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        l2.addWidget(self.rb_patch)

        # Container for Patch Controls
        self.container_patch = QWidget()
        lay_patch = QVBoxLayout(self.container_patch)
        lay_patch.setContentsMargins(20, 0, 0, 10) # Indent left

        self.lbl_patch = QLabel("Step: Draw Target Box")
        lay_patch.addWidget(self.lbl_patch)

        btn_apply_patch = QPushButton("Apply Patch (Enter)")
        btn_apply_patch.clicked.connect(self.apply_patch)
        btn_apply_patch.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        lay_patch.addWidget(btn_apply_patch)

        btn_undo_patch = QPushButton("Undo Patch Action (Z)")
        btn_undo_patch.clicked.connect(self.undo_cut)   
        btn_undo_patch.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        lay_patch.addWidget(btn_undo_patch)
        
        self.container_patch.setVisible(False)
        l2.addWidget(self.container_patch)

        # ===============================================
        # MODE C: Occlusion Paint
        # ===============================================
        self.rb_occ = QRadioButton("Occlusion Paint (O)")
        self.rb_occ.toggled.connect(lambda: self.set_mode("occlusion"))
        self.rb_occ.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        l2.addWidget(self.rb_occ)

        # Container for Occlusion Controls
        self.container_occ = QWidget()
        lay_occ = QVBoxLayout(self.container_occ)
        lay_occ.setContentsMargins(20, 0, 0, 10)

        # 1. UNDO (Occlusion)
        btn_undo_occ = QPushButton("Undo Paint Action (Z)")
        btn_undo_occ.clicked.connect(self.undo_cut)
        btn_undo_occ.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        lay_occ.addWidget(btn_undo_occ)
        
        self.container_occ.setVisible(False)
        l2.addWidget(self.container_occ)

        g2.setLayout(l2)
        sb_layout.addWidget(g2)

        # --- 3. ADJUSTMENTS (Same as before) ---
        g5 = QGroupBox("3. Adjustments")
        l5 = QVBoxLayout()
        self.chk_flip = QCheckBox("Flip Horizontal"); self.chk_flip.toggled.connect(self.update_comp_params)
        self.chk_flip.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        l5.addWidget(self.chk_flip)
        
        # (Copy pasting the rest of your existing g5 code)
        self.chk_harmony = QCheckBox("Color Harmonization")
        self.chk_harmony.setChecked(True)
        self.chk_harmony.toggled.connect(self.update_comp_params)
        self.chk_harmony.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        l5.addWidget(self.chk_harmony)

        self.chk_custom_harmony = QCheckBox("Custom Harmony Source")
        self.chk_custom_harmony.setStyleSheet("margin-left: 20px;")
        self.chk_custom_harmony.toggled.connect(self.toggle_custom_harmony)
        self.chk_custom_harmony.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        l5.addWidget(self.chk_custom_harmony)

        btn_apply_harmony = QPushButton("Apply Color Source")
        btn_apply_harmony.setStyleSheet("margin-left: 20px;")
        btn_apply_harmony.clicked.connect(self.confirm_and_hide_selection)
        btn_apply_harmony.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        l5.addWidget(btn_apply_harmony)
        
        self.chk_harmony.toggled.connect(self.chk_custom_harmony.setEnabled)
        self.chk_harmony.toggled.connect(btn_apply_harmony.setEnabled)

        l5.addWidget(QLabel("Harmonization Strength:"))
        self.sld_harmony_strength = QSlider(Qt.Orientation.Horizontal); self.sld_harmony_strength.setRange(0, 100); self.sld_harmony_strength.setValue(60)
        self.sld_harmony_strength.valueChanged.connect(self.update_comp_params)
        l5.addWidget(self.sld_harmony_strength)

        self.chk_relight = QCheckBox("Global Relighting"); self.chk_relight.setChecked(True)
        self.chk_relight.toggled.connect(self.update_comp_params)
        self.chk_relight.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        l5.addWidget(self.chk_relight)

        l5.addWidget(QLabel("Light Angle:"))
        self.sld_angle = QSlider(Qt.Orientation.Horizontal); self.sld_angle.setRange(0, 360); self.sld_angle.setValue(45)
        self.sld_angle.valueChanged.connect(self.update_comp_params)
        l5.addWidget(self.sld_angle)

        l5.addWidget(QLabel("Brush Size:"))
        self.sld_brush = QSlider(Qt.Orientation.Horizontal); self.sld_brush.setRange(5, 100); self.sld_brush.setValue(25)
        self.sld_brush.valueChanged.connect(self.update_brush_from_slider)
        l5.addWidget(self.sld_brush)

        g5.setLayout(l5)
        sb_layout.addWidget(g5)

        # SAVE/EXIT
        btn_save = QPushButton("SAVE RESULT")
        btn_save.setStyleSheet("background-color: green; color: white; padding: 10px;")
        btn_save.clicked.connect(self.save_result)
        btn_save.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        sb_layout.addWidget(btn_save)

        btn_exit = QPushButton("EXIT")
        btn_exit.setStyleSheet("background-color: #D32F2F; color: white; padding: 10px; font-weight: bold;")
        btn_exit.clicked.connect(self.close)
        btn_exit.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        sb_layout.addWidget(btn_exit)

        sb_layout.addStretch()
        sidebar.setWidget(sb_widget)
        sidebar.setWidgetResizable(True)
        layout.addWidget(sidebar, stretch=1)

    def toggle_custom_harmony(self, checked):
        if checked:
            # Enable picking mode
            self.is_picking_harmony = True
            self.set_mode("normal") # Ensure we aren't in patch/occlusion mode
            print(">>> Draw a box on the Background to select color source.")
        else:
            # Disable
            self.is_picking_harmony = False
            self.custom_harmony_rect = None
            self.process_composition()
        self.canvas_bg.update()
    
    def confirm_and_hide_selection(self):
        if self.custom_harmony_rect:
            print(">>> Selection Confirmed. Hiding Box.")
            self.is_box_visible = False  # Hide the box
            self.canvas_bg.update()      # Force redraw to remove it
            self.process_composition()   # Ensure color is applied

    def load_defaults(self):
        try:
            self.fg_img_orig = cv2.imread("shoes.png")
            self.bg_img_orig = cv2.imread("sidewalk.png")
            if self.fg_img_orig is None: raise Exception
            if self.bg_img_orig is None: raise Exception
        except:
            self.fg_img_orig = np.zeros((400, 400, 3), np.uint8)
            cv2.circle(self.fg_img_orig, (200, 200), 100, (0, 0, 255), -1)
            self.bg_img_orig = np.zeros((600, 800, 3), np.uint8)
            self.bg_img_orig[:] = (50, 50, 50)
        
        self.occlusion_mask = np.zeros(self.bg_img_orig.shape[:2], dtype=np.uint8)
        self.pos_x, self.pos_y = self.bg_img_orig.shape[1]//2, self.bg_img_orig.shape[0]//2
        
        self.canvas_fg.update_display(self.fg_img_orig)
        self.process_composition()

    def set_mode(self, mode):
        self.patch_step = 0
        self.patch_target = None
        self.patch_source = None

        if hasattr(self, 'lbl_patch'):
            self.lbl_patch.setText("Step: Draw Target Box")
        
        if hasattr(self, 'container_normal'):
            self.container_normal.setVisible(mode == "normal")
        
        if hasattr(self, 'container_patch'):
            self.container_patch.setVisible(mode == "patch")

        if hasattr(self, 'container_occ'):
            self.container_occ.setVisible(mode == "occlusion")

        # 2. Update mode and refresh
        self.current_mode = mode
        self.process_composition()

    # ==========================
    # LOGIC: GRABCUT (FIXED)
    # ==========================
    def undo_cut(self):
        if not self.history:
            print(">>> Nothing left to undo.")
            return

        print(">>> Undoing last action...")
        state = self.history.pop()

        # --- IMPORTANT: Check which version of history we have ---
        # This handles both old history (if any exists) and new history
        if "gc_mask" in state:
            # NEW Format
            if state["gc_mask"] is None:
                self.mask = None
                self.gc_initialized = False
                self.bgdModel = np.zeros((1, 65), np.float64)
                self.fgdModel = np.zeros((1, 65), np.float64)
            else:
                self.mask = state["gc_mask"]
                self.bgdModel = state["gc_bgd"]
                self.fgdModel = state["gc_fgd"]
                self.gc_initialized = state["gc_init"]
            
            # Restore Global State (New features)
            if state["bg_img"] is not None: self.bg_img_orig = state["bg_img"]
            if state["occ_mask"] is not None: self.occlusion_mask = state["occ_mask"]
            self.pos_x, self.pos_y = state["transform_pos"]
            self.scale = state["transform_scale"]
            self.rotation = state["transform_rot"]
            self.flip_h = state["transform_flip"]

        else:
            # FALLBACK for Old Format (prevents the KeyError)
            if state.get("mask") is None:
                self.mask = None
                self.gc_initialized = False
                self.bgdModel = np.zeros((1, 65), np.float64)
                self.fgdModel = np.zeros((1, 65), np.float64)
            else:
                self.mask = state["mask"]
                self.bgdModel = state["bgd"]
                self.fgdModel = state["fgd"]
                self.gc_initialized = state["init"]

        # Force updates on both canvases
        self.update_cutout_from_mask() 
        self.process_composition()

    def perform_grabcut(self, qrect):
        # 1. Get Image Dimensions
        h_img, w_img = self.fg_img_orig.shape[:2]

        # 2. Strict Coordinate Clamping (Prevent Crash)
        # We calculate start and end points constrained to image bounds (0 to w_img)
        x_start = max(0, min(int(qrect.x()), w_img - 1))
        y_start = max(0, min(int(qrect.y()), h_img - 1))
        x_end = max(0, min(int(qrect.x() + qrect.width()), w_img))
        y_end = max(0, min(int(qrect.y() + qrect.height()), h_img))
        
        w = x_end - x_start
        h = y_end - y_start

        if w < 1 or h < 1: return

        self.save_state()

        # 3. INITIALIZATION STEP
        if not self.gc_initialized:
            self.mask = np.zeros((h_img, w_img), np.uint8)
            
            # --- FIX: SAFEGUARD FOR FULL IMAGE SELECTION ---
            # If rect touches the image border, shrink it by 1px.
            # This ensures GrabCut always has at least 1px of "Sure Background" to learn from.
            gx, gy, gw, gh = x_start, y_start, w, h
            
            if gx <= 0: gx = 1; gw -= 1
            if gy <= 0: gy = 1; gh -= 1
            if gx + gw >= w_img: gw -= 1
            if gy + gh >= h_img: gh -= 1
            
            # Only proceed if we still have a valid box after shrinking
            if gw > 0 and gh > 0:
                rect = (gx, gy, gw, gh)
                try:
                    cv2.grabCut(self.fg_img_orig, self.mask, rect, self.bgdModel, self.fgdModel, 5, cv2.GC_INIT_WITH_RECT)
                    self.gc_initialized = True
                except cv2.error as e:
                    print(f"GrabCut Init Error: {e}")
        
        # 4. REFINEMENT STEP (Growing)
        else:
            # Use the CLAMPED coordinates (x_start, ...) for slicing
            roi = self.mask[y_start:y_start+h, x_start:x_start+w]
            
            # Logic: Keep existing Definite Foreground (1), else set to Probable Foreground (3)
            self.mask[y_start:y_start+h, x_start:x_start+w] = np.where(
                roi == cv2.GC_FGD, cv2.GC_FGD, cv2.GC_PR_FGD
            )

            try:
                cv2.grabCut(self.fg_img_orig, self.mask, None, self.bgdModel, self.fgdModel, 5, cv2.GC_INIT_WITH_MASK)
            except cv2.error as e:
                print(f"GrabCut Refine Error: {e}")

        self.update_cutout_from_mask()
    
    def execute_brush_grabcut(self):
        if self.mask is None or self.fg_img_orig is None: return
        try:
            # --- A. expand painted areas using Flood Fill ---
            h, w = self.fg_img_orig.shape[:2]
            # flood_mask needs to be 2 pixels larger than the image
            flood_mask = np.zeros((h + 2, w + 2), np.uint8)
            
            # Find all definite foreground pixels as seed points
            seeds = np.column_stack(np.where(self.mask == cv2.GC_FGD))
            
            if len(seeds) > 0:
                # To avoid too many flood fills, sample seeds evenly
                step = max(1, len(seeds) // 15)
                for i in range(0, len(seeds), step):
                    y, x = seeds[i]
                    # loDiff/upDiff can be adjusted for sensitivity
                    cv2.floodFill(self.fg_img_orig, flood_mask, (x, y), 255, 
                                  loDiff=(20, 20, 20), upDiff=(20, 20, 20), 
                                  flags=4 | cv2.FLOODFILL_MASK_ONLY)

            # --- B. Process flood-filled mask ---
            # Convert flood_mask to binary: 255 where filled, 0 elsewhere
            refined_area = flood_mask[1:-1, 1:-1]
            self.mask[refined_area == 255] = cv2.GC_PR_FGD
            
            # --- C. Re-run GrabCut with updated mask ---
            # Reset models
            self.main.bgdModel = np.zeros((1, 65), np.float64)
            self.main.fgdModel = np.zeros((1, 65), np.float64)

            # --- D. Run GrabCut
            cv2.grabCut(self.fg_img_orig, self.mask, None, 
                        self.main.bgdModel, self.main.fgdModel, 5, cv2.GC_INIT_WITH_MASK)
            
            self.update_cutout_from_mask()
            print(">>> 拖曳結束：全區域擴張並重新計算完成")
            
        except Exception as e:
            print(f"Hybrid Error: {e}")

    def lock_selection(self):
        """Converts PR_FGD (3) to FGD (1) so they don't get removed easily."""
        if self.mask is not None:
            self.save_state()
            self.mask = np.where(self.mask == cv2.GC_PR_FGD, cv2.GC_FGD, self.mask)
            self.mask = np.where(self.mask == cv2.GC_PR_BGD, cv2.GC_BGD, self.mask)
            self.update_cutout_from_mask()
            print(">>> Selection Frozen/Locked.")
            
    def update_cutout_from_mask(self):
        if self.mask is None:
            # 1. Reset the display to the clean original image
            self.canvas_fg.update_display(self.fg_img_orig)
            
            # 2. Clear the cutout data
            self.fg_cutout = None
            self.fg_cut_mask = None
            
            # 3. Update the composition to remove the object from the right screen
            self.process_composition()
            return
        
        # Visualize on Left Canvas
        vis_img = self.fg_img_orig.copy()
        # Green for Definite (1), Pink for Probable (3)
        vis_img[self.mask == cv2.GC_FGD] = [0, 255, 0]
        vis_img[self.mask == cv2.GC_PR_FGD] = [255, 0, 255]
        self.canvas_fg.update_display(vis_img)

        # Create Cutout for Right Canvas
        mask2 = np.where((self.mask == 2) | (self.mask == 0), 0, 1).astype('uint8')
        ys, xs = np.where(mask2 == 1)
        if len(ys) > 0:
            y1, y2, x1, x2 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
            self.fg_cutout = self.fg_img_orig[y1:y2, x1:x2].copy()
            self.fg_cut_mask = mask2[y1:y2, x1:x2].copy()
        else:
            self.fg_cutout = None

        self.process_composition()

    # ==========================
    # LOGIC: PATCH TOOL
    # ==========================
    def register_patch_rect(self, qrect):
        rect = (int(qrect.x()), int(qrect.y()), int(qrect.width()), int(qrect.height()))
        if self.patch_step == 0:
            self.patch_target = qrect
            self.patch_step = 1
            self.lbl_patch.setText("Step: Draw Source (Copy From)")
        elif self.patch_step == 1:
            self.patch_source = qrect
            self.patch_step = 2
            self.lbl_patch.setText("Step: Press 'Apply Patch'")

    def apply_patch(self):
        # Verify we are in the correct step and have an image
        if self.patch_step == 2 and self.bg_img_orig is not None:
            
            # Get Image Dimensions
            h_img, w_img = self.bg_img_orig.shape[:2]

            # --- HELPER: CLAMP RECT TO IMAGE BOUNDS ---
            def get_safe_coords(qrect):
                x, y = int(qrect.x()), int(qrect.y())
                w, h = int(qrect.width()), int(qrect.height())
                
                # 1. Clamp Top-Left
                x = max(0, min(x, w_img - 1))
                y = max(0, min(y, h_img - 1))
                
                # 2. Clamp Size
                w = min(w, w_img - x)
                h = min(h, h_img - y)
                
                # 3. Safety shrink for edge cases
                if x + w >= w_img: w -= 1
                if y + h >= h_img: h -= 1
                if x == 0: x += 1; w -= 1
                if y == 0: y += 1; h -= 1
                
                return x, y, w, h

            # 1. Get Safe Coordinates
            tx, ty, tw, th = get_safe_coords(self.patch_target)
            sx, sy, sw, sh = get_safe_coords(self.patch_source)
            
            # 2. Check dimensions
            if tw <= 1 or th <= 1 or sw <= 1 or sh <= 1:
                print("Error: Selection too small or too close to edge.")
                return

            # 3. Extract Source
            src_patch = self.bg_img_orig[sy:sy+sh, sx:sx+sw]
            
            try:
                # 4. Resize Source
                src_patch = cv2.resize(src_patch, (tw, th))
                
                # 5. Create Mask
                mask = 255 * np.ones(src_patch.shape, src_patch.dtype)
                
                # 6. Calculate Center
                center = (int(tx + tw // 2), int(ty + th // 2))

                # --- CORRECTED SECTION START ---
                self.save_state() # Save Undo state right before modifying
                
                # Apply Clone (Only once)
                self.bg_img_orig = cv2.seamlessClone(
                    src_patch, self.bg_img_orig, mask, center, cv2.NORMAL_CLONE
                )
                # --- CORRECTED SECTION END ---

                print("Patch Applied Successfully.")

                # Reset State
                self.patch_target = None
                self.patch_source = None
                self.patch_step = 0
                self.lbl_patch.setText("Step: Draw Target Box")
                self.process_composition()

            except cv2.error as e:
                print(f"Clone Error: {e}")

    # ==========================
    # LOGIC: COMPOSITING
    # ==========================

    def update_brush_from_slider(self):
        self.brush_size = self.sld_brush.value()
        # Trigger update on canvas to redraw cursor circle
        self.canvas_bg.update()

    def process_composition(self):
        if self.bg_img_orig is None: return

        # Reset geometry info
        self.active_geom = None

        # 1. Background Base
        comp = self.bg_img_orig.copy()
        self.current_mask_result = np.zeros(comp.shape[:2], dtype=np.uint8)
        # -----------------------------------------

        # 2. Global Lighting (Background)
        if self.chk_relight.isChecked():
            angle = self.sld_angle.value()
            comp = ImageUtils.apply_light_source(comp, angle)

        # 3. Place Foreground
        if self.fg_cutout is not None:
            fg = self.fg_cutout.copy()
            mask = self.fg_cut_mask.copy()

            # Transform
            h, w = fg.shape[:2]
            nw, nh = int(w * self.scale), int(h * self.scale)
            if nw > 0 and nh > 0:
                fg = cv2.resize(fg, (nw, nh))
                mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_LINEAR) 
                
                if self.flip_h: 
                    fg = cv2.flip(fg, 1); mask = cv2.flip(mask, 1)
                fg = ImageUtils.rotate_image(fg, self.rotation)
                mask = ImageUtils.rotate_image(mask, self.rotation)

                # A. ERODE: Shave off 1 pixel to remove the white halo
                # If the halo is thick, change iterations to 2
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.erode(mask, kernel, iterations=1) 

                # B. BLUR: Soften the new edge so it doesn't look like a sticker
                mask = mask.astype(np.float32) # Convert to float for blending
                if mask.max() > 1.0: mask /= 255.0 
                
                # Gaussian Blur (Kernel size 3x3 or 5x5 is good)
                mask = cv2.GaussianBlur(mask, (3, 3), 0)
                
                # Clip to ensure valid alpha range
                mask = np.clip(mask, 0, 1)

                # Positioning
                fh, fw = fg.shape[:2]
                tx = int(self.pos_x - fw/2)
                ty = int(self.pos_y - fh/2)

                # We store: Center X, Center Y, Width, Height
                self.active_geom = (self.pos_x, self.pos_y, fw, fh)
                
                # Bounds
                bh, bw = comp.shape[:2]
                x1, y1 = max(0, tx), max(0, ty)
                x2, y2 = min(bw, tx+fw), min(bh, ty+fh)

                if x2 > x1 and y2 > y1:
                    sx, sy = x1 - tx, y1 - ty
                    fg_crop = fg[sy:sy+(y2-y1), sx:sx+(x2-x1)]
                    mask_crop = mask[sy:sy+(y2-y1), sx:sx+(x2-x1)]
                    bg_crop = comp[y1:y2, x1:x2]

                    # Color Harmony
                    fg_blend = fg_crop.copy()
                    
                    # DEBUG PRINT 1
                    if not self.chk_harmony.isChecked():
                        print(">> SKIPPING: Main Color Harmonization Checkbox is OFF")
                    
                    if self.chk_harmony.isChecked():
                        try:
                            # Determine which background reference to use
                            target_ref = bg_crop.copy() # Default: area behind object
                            
                            # DEBUG PRINT 2
                            if self.chk_custom_harmony.isChecked():
                                if self.custom_harmony_rect is None:
                                    print(">> CUSTOM FAIL: Checkbox ON, but Rect is None (Did you draw it?)")
                                else:
                                    print(f">> CUSTOM OK: Using Rect {self.custom_harmony_rect}")
                            
                            if (self.chk_custom_harmony.isChecked() and 
                                self.custom_harmony_rect is not None):
                                # Use Custom Region
                                cx, cy, cw, ch = self.custom_harmony_rect
                                
                                # Bounds Check
                                cx = max(0, min(int(cx), comp.shape[1]-1))
                                cy = max(0, min(int(cy), comp.shape[0]-1))
                                cw = min(int(cw), comp.shape[1]-cx)
                                ch = min(int(ch), comp.shape[0]-cy)
                                
                                if cw > 0 and ch > 0:
                                    target_ref = self.bg_img_orig[cy:cy+ch, cx:cx+cw].copy()
                                else:
                                    print(">> CUSTOM FAIL: Rect size is 0 after bounds check")

                            # Use Slider Strength
                            strength = self.sld_harmony_strength.value() / 100.0

                            if target_ref.size > 0 and fg_blend.size > 0:
                                harmonized = ImageUtils.match_color_stats(fg_crop, target_ref)
                                fg_blend = cv2.addWeighted(fg_crop, (1.0 - strength), harmonized, strength, 0)
                        except Exception as e: 
                            print(f"Harmony Error: {e}")
                    
                    # Lighting on Foreground
                    if self.chk_relight.isChecked():
                        fg_blend = ImageUtils.apply_light_source(fg_blend, self.sld_angle.value())

                    # Occlusion Logic
                    occ_crop = self.occlusion_mask[y1:y2, x1:x2]
                    alpha = mask_crop 
                    
                    visibility = 1.0 - (occ_crop.astype(float) / 255.0)
                    
                    # Calculate Final Alpha (Shape: HxWx1)
                    final_alpha_1ch = alpha * visibility
                    
                    # --- NEW: Save this to the Full Mask ---
                    # Convert 0.0-1.0 to 0-255
                    self.current_mask_result[y1:y2, x1:x2] = (final_alpha_1ch * 255).astype(np.uint8)
                    # ---------------------------------------

                    # Blend Composite
                    final_alpha_3ch = np.repeat(final_alpha_1ch[:, :, None], 3, axis=2)
                    comp[y1:y2, x1:x2] = (fg_blend * final_alpha_3ch + bg_crop * (1.0 - final_alpha_3ch)).astype(np.uint8)

        self.current_result = comp
        
        # Occlusion Visualization Overlay (Red)
        if self.current_mode == "occlusion":
            red_overlay = np.zeros_like(comp)
            red_overlay[:,:,2] = 255
            
            # Safe check: Ensure mask exists
            if self.occlusion_mask is not None:
                occ_bool = self.occlusion_mask > 0
                
                if np.any(occ_bool):
                    comp[occ_bool] = cv2.addWeighted(comp[occ_bool], 0.7, red_overlay[occ_bool], 0.3, 0)

        self.canvas_bg.update_display(comp)

    # --- Interaction Helpers ---
    def load_fg(self):
        p, _ = QFileDialog.getOpenFileName(self, "Load FG")
        if p:
            img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
            
            if img is None:
                print("Error: Could not load foreground image.")
                return
            
            # Automatic Old Photo Detection and Repair
            if ImageUtils.is_old_photo(img):
                reply = QMessageBox.question(
                    self, 'Determining whether it is an old photograph', 
                    "If a photo has cracks or looks faded, should we automatically fix and enhance it?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                
                # If user agrees, perform repair
                if reply == QMessageBox.StandardButton.Yes:
                    img = ImageUtils.repair_old_photo(img)
                    print(">>> Foreground has been automatically restored and enhanced using CLAHE")
            
            self.fg_img_orig = img
            self.gc_initialized = False 
            self.mask = None
            self.canvas_fg.update_display(self.fg_img_orig)
            
            self.fg_cutout = None
            self.process_composition()

    def load_bg(self):
        p, _ = QFileDialog.getOpenFileName(self, "Load BG")
        if p:
            img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
            
            if img is None:
                print("Error: Could not load background image.")
                return

            # Automatic Old Photo Detection and Repair
            if ImageUtils.is_old_photo(img):
                reply = QMessageBox.question(
                    self, 'Determining whether it is an old photograph', 
                    "If a photo has cracks or looks faded, should we automatically fix and enhance it?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    img = ImageUtils.repair_old_photo(img)
                    print(">>> >>> Background has been automatically restored and enhanced using CLAHE")

            self.bg_img_orig = img
            # Reset occlusion mask and position
            self.occlusion_mask = np.zeros(self.bg_img_orig.shape[:2], dtype=np.uint8)
            self.pos_x, self.pos_y = self.bg_img_orig.shape[1]//2, self.bg_img_orig.shape[0]//2
            
            self.process_composition()

    def update_comp_params(self):
        self.flip_h = self.chk_flip.isChecked()
        self.process_composition()

    def start_dragging_object(self, x, y):
        self.is_dragging_obj = True
        self.drag_offset = (self.pos_x - x, self.pos_y - y)
    
    def drag_object(self, x, y):
        if self.is_dragging_obj:
            self.pos_x = x + self.drag_offset[0]
            self.pos_y = y + self.drag_offset[1]
            self.process_composition()
            
    def stop_dragging(self):
        self.is_dragging_obj = False

    def paint_occlusion(self, x, y, is_erasing, is_new_stroke=False):
        if is_new_stroke:
             self.save_state()

        color = 0 if is_erasing else 255
        cv2.circle(self.occlusion_mask, (x, y), self.brush_size, color, -1)
        self.process_composition()
    
    def update_brush_size(self, change):
        self.brush_size = max(5, min(200, self.brush_size + change))

    def update_scale_scroll(self, change):
        self.sld_scale.setValue(self.sld_scale.value() + int(change*100))

    def save_result(self):
        if hasattr(self, 'current_result') and self.current_result is not None:
            # 1. Open File Dialog
            path, _ = QFileDialog.getSaveFileName(self, "Save Result", "result.jpg", "Images (*.jpg *.png)")
            
            if path:
                
                # 2. Temporarily switch mode to remove Red Overlay (if in occlusion mode)
                temp_mode = self.current_mode
                self.current_mode = "move"
                self.process_composition() # Refresh cleanly
                
                # 3. Save the Composite Image
                cv2.imwrite(path, self.current_result)
                print(f"Saved Composite: {path}")

                # 4. Save the Mask Image
                if hasattr(self, 'current_mask_result') and self.current_mask_result is not None:
                    # Create mask filename (e.g. "photo.jpg" -> "photo_mask.png")
                    # We use PNG for masks because it is lossless (no blurry edges)
                    root, ext = os.path.splitext(path)
                    mask_path = f"{root}_mask.png"
                    
                    cv2.imwrite(mask_path, self.current_mask_result)
                    print(f"Saved Mask: {mask_path}")

                # 5. Restore original mode
                self.current_mode = temp_mode
                self.process_composition()

    # ==========================
    # KEYBOARD SHORTCUTS
    # ==========================
    def keyPressEvent(self, event):
        # 1. Patch Tool (Key: R)
        if event.key() == Qt.Key.Key_R:
            if self.current_mode == "patch":
                # If already in Patch mode, switch back to Normal
                self.rb_normal.setChecked(True)
                print(">>> Toggled OFF Patch Tool (Back to Standard)")
            else:
                # Otherwise, switch to Patch mode
                self.rb_patch.setChecked(True)
                print(">>> Switched to Patch Tool")

        # 2. Occlusion Paint (Key: O)
        elif event.key() == Qt.Key.Key_O:
            if self.current_mode == "occlusion":
                # If already in Occlusion mode, switch back to Normal
                self.rb_normal.setChecked(True)
                print(">>> Toggled OFF Occlusion Paint (Back to Standard)")
            else:
                # Otherwise, switch to Occlusion mode
                self.rb_occ.setChecked(True)
                print(">>> Switched to Occlusion Paint")

        # 3. Lock Selection (Key: Space)
        elif event.key() == Qt.Key.Key_Space:
            if self.current_mode == "normal":
                self.lock_selection()
            else:
                print(">>> Lock only works in Cut Mode")

        # 4. Apply Patch (Key: Enter or Return)
        elif event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            if self.current_mode == "patch":
                self.apply_patch()  
            elif self.chk_custom_harmony.isChecked():
                print(">>> Applying Custom Harmony...")
                self.confirm_and_hide_selection()
        
        # Optional: 'Z' for Undo (already defined in button, but good to map here too)
        elif event.key() == Qt.Key.Key_Z:          
            self.undo_cut()
        
        if event.key() == Qt.Key.Key_N:
            if self.mask is not None:
                self.run_grabcut_iteration()
            else:
                print(">>> Fixed the mask after Select ROI")

        # Pass other events (like standard system shortcuts) up the chain
        else:
            super().keyPressEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
