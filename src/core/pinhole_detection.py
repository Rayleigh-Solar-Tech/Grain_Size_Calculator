#!/usr/bin/env python3
"""
Pinhole Detection Module for Grain Size Calculator
Integrates SAM-based pinhole detection with user confirmation.
"""

import os
import csv
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from ultralytics import SAM

class PinholeDetector:
    """SAM-based pinhole detector with configurable parameters."""
    
    def __init__(self):
        # Configuration parameters
        self.config = {
            # Seed (pre-SAM) settings
            'BH_KSIZE': 31,
            'PCT_DARKEST': 8.0,
            'BH_TOP_PCT': 92.0,
            'PAD_BOX': 5,
            'NMS_IOU': 0.2,
            
            # Post-filter settings
            'MEAN_MAX_PCT': 18.0,
            'DARK_MAX_MEAN_NORM': 0.24,
            'MIN_EQUIV_DIAM_UM': 0.20,
            
            # Shape filters
            'USE_SHAPE_FILTERS': False,
            'MIN_CIRC': 0.45,
            'MAX_ECC': 0.92,
            'MIN_AREA_PX': 3,
            
            # Bright-but-round override
            'BRIGHT_ROUND_OVERRIDE': True,
            'MIN_CIRC_OVERRIDE': 0.70,
            'MAX_ECC_OVERRIDE': 0.70,
            'BRIGHT_ROUND_MAX_NORM': 0.55,
            'BRIGHT_ROUND_MAX_PCT': 70.0,
            
            # Rescue settings
            'RESCUE_ADAPT_BLOCK': 35,
            'RESCUE_ADAPT_C': 2,
            'RESCUE_OPEN': 3,
            
            # ROI settings
            'IGNORE_BOTTOM_ONLY': True,
            'BOTTOM_BANNER_PCT': 0.08,
            'ROI_KEEP_FRACTION': 0.60,
        }
    
    def to_gray_2d(self, im):
        """Convert image to 2D grayscale."""
        if im.ndim == 3:
            if im.shape[2] >= 3:
                return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            return im[..., 0]
        return im
    
    def normalize01(self, a):
        """Normalize array to 0-1 range."""
        a = a.astype(np.float32)
        rng = float(a.max() - a.min()) or 1.0
        return (a - a.min()) / rng
    
    def nms_xyxy(self, boxes, iou_th=0.3):
        """Non-maximum suppression for bounding boxes."""
        if not len(boxes):
            return []
        b = np.asarray(boxes, dtype=float)
        x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = np.argsort((x2 - x1) * (y2 - y1))  # small-first
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
            order = order[1:][iou <= iou_th]
        return b[keep].astype(int).tolist()
    
    def circularity(self, cnt):
        """Calculate circularity of contour."""
        A = cv2.contourArea(cnt)
        P = cv2.arcLength(cnt, True)
        return (4 * math.pi * A / (P * P)) if P > 0 else 0.0
    
    def eccentricity(self, cnt):
        """Calculate eccentricity of contour."""
        if len(cnt) < 5:
            return 1.0
        (_, _), (MA, ma), _ = cv2.fitEllipse(cnt)
        MA, ma = max(MA, ma), min(MA, ma)
        if MA <= 1e-6:
            return 1.0
        return float(np.sqrt(1.0 - (ma / MA) ** 2))
    
    def masked_mean(self, arr, mask):
        """Calculate mean of array where mask is True."""
        return float(arr[mask].mean()) if mask.any() else 1e9
    
    def build_bottom_roi(self, h, w):
        """Build ROI mask ignoring bottom banner."""
        if not self.config['IGNORE_BOTTOM_ONLY']:
            return np.ones((h, w), dtype=bool)
        m = np.ones((h, w), dtype=bool)
        cut = int(round(self.config['BOTTOM_BANNER_PCT'] * h))
        if cut > 0:
            m[h - cut:h, :] = False
        return m
    
    def filter_and_measure(self, m_bool, gray, gray_norm, um_per_px, roi_mask):
        """Filter and measure a binary mask."""
        A = int(m_bool.sum())
        if A < max(1, self.config['MIN_AREA_PX']):
            return False, {"A": A}, "tiny_area"
        
        if roi_mask is not None:
            in_roi = int((m_bool & roi_mask).sum())
            frac_in = in_roi / float(max(1, A))
            if frac_in < self.config['ROI_KEEP_FRACTION']:
                return False, {"A": A}, "out_of_roi"
        
        m_u8 = (m_bool.astype(np.uint8) * 255)
        cnts, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return False, {"A": A}, "no_contour"
        cnt = max(cnts, key=cv2.contourArea)
        
        mean_raw = self.masked_mean(gray, m_bool)
        mean_norm = self.masked_mean(gray_norm, m_bool)
        
        equiv_diam_px = math.sqrt(4 * A / math.pi)
        equiv_diam_um = equiv_diam_px * um_per_px
        if equiv_diam_um < self.config['MIN_EQUIV_DIAM_UM']:
            return False, {"A": A}, "too_small"
        
        circ = self.circularity(cnt)
        ecc = self.eccentricity(cnt)
        
        if self.config['USE_SHAPE_FILTERS'] and (circ < self.config['MIN_CIRC'] or ecc > self.config['MAX_ECC']):
            return False, {"A": A}, "shape_fail"
        
        # Darkness gates
        mean_max_val = np.percentile(gray, self.config['MEAN_MAX_PCT'])
        too_bright = (mean_raw > mean_max_val) or (mean_norm > self.config['DARK_MAX_MEAN_NORM'])
        
        if too_bright:
            # Bright-but-round override
            not_too_bright = (mean_norm <= self.config['BRIGHT_ROUND_MAX_NORM']) and \
                           (mean_raw <= np.percentile(gray, self.config['BRIGHT_ROUND_MAX_PCT']))
            if self.config['BRIGHT_ROUND_OVERRIDE'] and not_too_bright and \
               (circ >= self.config['MIN_CIRC_OVERRIDE']) and (ecc <= self.config['MAX_ECC_OVERRIDE']):
                return True, {"A": A, "eq_um": equiv_diam_um, "mean_raw": mean_raw,
                            "mean_norm": mean_norm, "circ": circ, "ecc": ecc, "cnt": cnt}, "kept_bright_round"
            return False, {"A": A}, "too_bright"
        
        return True, {"A": A, "eq_um": equiv_diam_um, "mean_raw": mean_raw,
                     "mean_norm": mean_norm, "circ": circ, "ecc": ecc, "cnt": cnt}, "kept"
    
    def detect_pinholes(self, image_path, frame_width_um, model_gpu="sam_l.pt", model_cpu="sam_l.pt"):
        """
        Main pinhole detection method.
        
        Returns:
            dict: {
                'count': int,
                'pinholes': list of pinhole data,
                'preview_image': np.array (BGR overlay),
                'success': bool,
                'message': str
            }
        """
        try:
            # Load image
            im = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if im is None:
                return {'success': False, 'message': f"Could not read image: {image_path}"}
            
            gray = self.to_gray_2d(im)
            H, W = gray.shape
            um_per_px = frame_width_um / float(W)
            gray_norm = self.normalize01(gray)
            roi_mask = self.build_bottom_roi(H, W)
            
            # Create seeds using blackhat
            ks = self.config['BH_KSIZE'] | 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
            bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
            
            raw_thr = np.percentile(gray, self.config['PCT_DARKEST'])
            bh_thr = np.percentile(bh, self.config['BH_TOP_PCT'])
            cand = (gray <= raw_thr) & (bh >= bh_thr)
            cand &= roi_mask
            
            # Create seed boxes
            cand_u8 = (cand.astype(np.uint8) * 255)
            dist = cv2.distanceTransform(255 - cand_u8, cv2.DIST_L2, 3)
            dil = cv2.dilate(dist, np.ones((3, 3), np.uint8))
            peaks = (dist == dil) & (cand)
            peaks_u8 = peaks.astype(np.uint8) * 255
            
            contours, _ = cv2.findContours(peaks_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            seed_boxes = []
            for c in contours:
                if cv2.contourArea(c) < 1:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                x1 = max(0, x - self.config['PAD_BOX'])
                y1 = max(0, y - self.config['PAD_BOX'])
                x2 = min(W - 1, x + w - 1 + self.config['PAD_BOX'])
                y2 = min(H - 1, y + h - 1 + self.config['PAD_BOX'])
                seed_boxes.append([x1, y1, x2, y2])
            seed_boxes = self.nms_xyxy(seed_boxes, self.config['NMS_IOU'])
            
            # Create overlay for preview
            overlay = cv2.cvtColor((self.normalize01(gray) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            
            pinholes = []
            
            # Try SAM detection if we have seeds
            if len(seed_boxes) > 0:
                try:
                    device = "cuda:0" if torch.cuda.is_available() else "cpu"
                    sam = SAM(model_gpu if device.startswith("cuda") else model_cpu)
                    rgb8 = np.ascontiguousarray((self.normalize01(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)) * 255).astype(np.uint8))
                    
                    res = sam.predict(source=rgb8, bboxes=seed_boxes, device=device, save=False, verbose=False, half=device.startswith("cuda"))
                    masks = getattr(res[0], "masks", None)
                    
                    if masks is not None and getattr(masks, "data", None) is not None:
                        try:
                            m_np = res[0].masks.data.detach().cpu().numpy()
                        except Exception:
                            m_np = res[0].masks.data.cpu().numpy()
                        
                        kept = 0
                        for i in range(m_np.shape[0]):
                            m_bool = (m_np[i] > 0.5)
                            kept_flag, metr, reason = self.filter_and_measure(m_bool, gray, gray_norm, um_per_px, roi_mask)
                            
                            if not kept_flag:
                                continue
                            
                            cnt_keep = metr["cnt"]
                            M = cv2.moments(cnt_keep)
                            cx = int(M["m10"] / M["m00"]) if M["m00"] else int(np.mean(np.where(m_bool)[1]))
                            cy = int(M["m01"] / M["m00"]) if M["m00"] else int(np.mean(np.where(m_bool)[0]))
                            A = metr["A"]
                            eq_um = metr["eq_um"]
                            eq_px = (eq_um / um_per_px)
                            area_um2 = A * (um_per_px ** 2)
                            circ = metr.get("circ", 1.0)
                            ecc = metr.get("ecc", 0.0)
                            
                            kept += 1
                            
                            # Draw on overlay
                            cv2.drawContours(overlay, [cnt_keep], -1, (0, 255, 0), 1, cv2.LINE_AA)
                            cv2.circle(overlay, (cx, cy), 2, (0, 255, 0), -1, cv2.LINE_AA)
                            txt = f"{eq_um:.2f} µm"
                            cv2.putText(overlay, txt, (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
                            cv2.putText(overlay, txt, (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                            
                            if reason == "kept_bright_round":
                                cv2.drawContours(overlay, [cnt_keep], -1, (255, 255, 0), 1, cv2.LINE_AA)
                                cv2.putText(overlay, "bright-round", (cx, max(12, cy - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
                                cv2.putText(overlay, "bright-round", (cx, max(12, cy - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                            
                            pinholes.append({
                                "pinhole_id": kept,
                                "cx_px": cx,
                                "cy_px": cy,
                                "area_px": int(A),
                                "equiv_diam_px": float(eq_px),
                                "mean_intensity": float(metr["mean_raw"]),
                                "circularity": float(circ),
                                "eccentricity": float(ecc),
                                "area_um2": float(area_um2),
                                "equiv_diam_um": float(eq_um)
                            })
                
                except Exception as e:
                    # Fall back to rescue method if SAM fails
                    pass
            
            # Rescue method if no pinholes found
            if len(pinholes) == 0:
                # Simple adaptive threshold approach
                th = cv2.adaptiveThreshold(
                    cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                    255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                    self.config['RESCUE_ADAPT_BLOCK'] | 1, self.config['RESCUE_ADAPT_C'])
                
                if self.config['RESCUE_OPEN'] > 0:
                    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.config['RESCUE_OPEN'] | 1, self.config['RESCUE_OPEN'] | 1))
                    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k)
                
                th = (th > 0).astype(np.uint8) * 255
                th[~roi_mask] = 0
                
                cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                kept = 0
                for cnt in cnts:
                    mask = np.zeros_like(gray, np.uint8)
                    cv2.drawContours(mask, [cnt], -1, 255, -1)
                    m_bool = mask > 0
                    kept_flag, metr, reason = self.filter_and_measure(m_bool, gray, gray_norm, um_per_px, roi_mask)
                    
                    if not kept_flag:
                        continue
                    
                    cnt_keep = metr["cnt"]
                    M = cv2.moments(cnt_keep)
                    cx = int(M["m10"] / M["m00"]) if M["m00"] else int(np.mean(np.where(m_bool)[1]))
                    cy = int(M["m01"] / M["m00"]) if M["m00"] else int(np.mean(np.where(m_bool)[0]))
                    A = metr["A"]
                    eq_um = metr["eq_um"]
                    eq_px = (eq_um / um_per_px)
                    area_um2 = A * (um_per_px ** 2)
                    circ = metr.get("circ", 1.0)
                    ecc = metr.get("ecc", 0.0)
                    
                    kept += 1
                    
                    # Draw on overlay
                    cv2.drawContours(overlay, [cnt_keep], -1, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.circle(overlay, (cx, cy), 2, (0, 255, 0), -1, cv2.LINE_AA)
                    txt = f"{eq_um:.2f} µm"
                    cv2.putText(overlay, txt, (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(overlay, txt, (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                    
                    pinholes.append({
                        "pinhole_id": kept,
                        "cx_px": cx,
                        "cy_px": cy,
                        "area_px": int(A),
                        "equiv_diam_px": float(eq_px),
                        "mean_intensity": float(metr["mean_raw"]),
                        "circularity": float(circ),
                        "eccentricity": float(ecc),
                        "area_um2": float(area_um2),
                        "equiv_diam_um": float(eq_um)
                    })
            
            return {
                'success': True,
                'count': len(pinholes),
                'pinholes': pinholes,
                'preview_image': overlay,
                'message': f"Detected {len(pinholes)} pinholes"
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"Error during pinhole detection: {str(e)}",
                'count': 0,
                'pinholes': [],
                'preview_image': None
            }
    
    def save_pinhole_csv(self, pinholes, output_path, pinhole_count_override=None):
        """Save pinhole data to CSV file."""
        try:
            # Use override count if provided, otherwise use actual detected count
            final_count = pinhole_count_override if pinhole_count_override is not None else len(pinholes)
            
            # Create summary row
            summary_data = {
                'total_pinholes_detected': len(pinholes),
                'final_pinhole_count': final_count,
                'user_modified': pinhole_count_override is not None
            }
            
            # Save detailed pinhole data
            if pinholes:
                fieldnames = [
                    "pinhole_id", "cx_px", "cy_px", "area_px", "equiv_diam_px",
                    "mean_intensity", "circularity", "eccentricity", "area_um2", "equiv_diam_um"
                ]
                
                with open(output_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(pinholes)
            
            # Save summary file
            summary_path = output_path.replace('.csv', '_summary.csv')
            with open(summary_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=['metric', 'value'])
                writer.writeheader()
                for key, value in summary_data.items():
                    writer.writerow({'metric': key, 'value': value})
            
            return True, f"Pinhole data saved to {output_path}"
            
        except Exception as e:
            return False, f"Error saving pinhole CSV: {str(e)}"


def create_pinhole_detector():
    """Factory function to create pinhole detector."""
    return PinholeDetector()