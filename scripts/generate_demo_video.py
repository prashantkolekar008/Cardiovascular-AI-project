"""
CardioVision 3D - Automated Demo Video Generator
Renders a smooth 1080p 60-second product demo video with:
- Cinematic titles & tech stack badges
- Ken-Burns pan/zoom camera physics
- Simulated cursor interactions and click ripples
- Lower-third clinical explanations
- Preset transitions (High-Risk Case 03 -> Low-Risk Case 01 -> Vitals Simulation)
- Closing architecture summary
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WIDTH = 1920
HEIGHT = 1080
FPS = 30

BASE_DIR = r"C:\Users\Manas95\.gemini\antigravity\scratch\Cardiovascular-AI-project"
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "docs", "screenshots")
OUTPUT_VIDEO_PATH = r"C:\Users\Manas95\Downloads\CardioVision_3D_Demo.mp4"
REPO_VIDEO_PATH = os.path.join(BASE_DIR, "docs", "CardioVision_3D_Demo.mp4")

# Load base screenshots
img_case03 = cv2.imread(os.path.join(SCREENSHOTS_DIR, "cardiovision_case03_high_risk.png"))
img_case01 = cv2.imread(os.path.join(SCREENSHOTS_DIR, "cardiovision_case01_low_risk.png"))
img_modal  = cv2.imread(os.path.join(SCREENSHOTS_DIR, "cardiovision_vitals_modal.png"))

# Fonts
FONT_BOLD = "C:/Windows/Fonts/segoeuib.ttf" if os.path.exists("C:/Windows/Fonts/segoeuib.ttf") else "C:/Windows/Fonts/arialbd.ttf"
FONT_REG  = "C:/Windows/Fonts/segoeui.ttf" if os.path.exists("C:/Windows/Fonts/segoeui.ttf") else "C:/Windows/Fonts/arial.ttf"
FONT_MONO = "C:/Windows/Fonts/consola.ttf" if os.path.exists("C:/Windows/Fonts/consola.ttf") else "C:/Windows/Fonts/arial.ttf"

def get_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except:
        return ImageFont.load_default()

font_title = get_font(FONT_BOLD, 52)
font_subtitle = get_font(FONT_REG, 26)
font_badge = get_font(FONT_BOLD, 18)
font_body = get_font(FONT_REG, 22)
font_banner_h1 = get_font(FONT_BOLD, 22)
font_banner_sub = get_font(FONT_REG, 17)

def draw_lower_third(frame_bgr, title, subtitle="", tag="CLINICAL DEMO"):
    """Draws a sleek clinical DICOM lower-third glass banner."""
    pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img, "RGBA")
    
    x0, y0, x1, y1 = 60, 920, 1860, 1020
    # Glass backdrop
    draw.rounded_rectangle([x0, y0, x1, y1], radius=16, fill=(10, 16, 28, 225), outline=(34, 211, 238, 140), width=2)
    # Dynamic accent pill
    tag_w = int(draw.textlength(tag, font=font_badge)) if hasattr(draw, "textlength") else len(tag) * 11
    badge_w = tag_w + 28
    draw.rounded_rectangle([x0 + 20, y0 + 16, x0 + 20 + badge_w, y0 + 46], radius=8, fill=(14, 165, 233, 200))
    draw.text((x0 + 34, y0 + 20), tag, font=font_badge, fill=(255, 255, 255, 255))
    
    # Title & Subtitle offset safely
    tx = x0 + 20 + badge_w + 18
    draw.text((tx, y0 + 18), title, font=font_banner_h1, fill=(255, 255, 255, 255))
    if subtitle:
        draw.text((tx, y0 + 54), subtitle, font=font_banner_sub, fill=(148, 163, 184, 255))
        
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def draw_cursor(frame_bgr, x, y, click_progress=0.0):
    """Draws simulated cursor and animated click ripple."""
    h, w = frame_bgr.shape[:2]
    cx, cy = int(x), int(y)
    
    if 0 <= cx < w and 0 <= cy < h:
        # Click ripple
        if click_progress > 0:
            radius = int(8 + 35 * click_progress)
            alpha = max(0, 1.0 - click_progress)
            overlay = frame_bgr.copy()
            cv2.circle(overlay, (cx, cy), radius, (238, 211, 34), 3) # Cyan
            cv2.addWeighted(overlay, alpha * 0.7, frame_bgr, 1.0 - (alpha * 0.7), 0, frame_bgr)
            
        # Draw sleek arrow cursor
        pts = np.array([[cx, cy], [cx, cy + 24], [cx + 6, cy + 18], [cx + 14, cy + 26], [cx + 18, cy + 24], [cx + 10, cy + 16], [cx + 18, cy + 16]], np.int32)
        cv2.fillPoly(frame_bgr, [pts], (255, 255, 255))
        cv2.polylines(frame_bgr, [pts], True, (20, 20, 20), 2)
    return frame_bgr

def crop_and_scale(img, cx, cy, zoom):
    """Extracts a viewport centered at (cx, cy) with zoom factor, scaled to 1920x1080."""
    h, w = img.shape[:2]
    crop_w = int(w / zoom)
    crop_h = int(h / zoom)
    
    x1 = max(0, min(w - crop_w, int(cx - crop_w / 2)))
    y1 = max(0, min(h - crop_h, int(cy - crop_h / 2)))
    x2 = x1 + crop_w
    y2 = y1 + crop_h
    
    cropped = img[y1:y2, x1:x2]
    return cv2.resize(cropped, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)

def create_title_card(progress=1.0):
    """Generates an aesthetic opening title card frame."""
    pil_img = Image.new("RGBA", (WIDTH, HEIGHT), (6, 9, 15, 255))
    draw = ImageDraw.Draw(pil_img, "RGBA")
    
    # Subtle background medical grid
    for gx in range(0, WIDTH, 80):
        draw.line([(gx, 0), (gx, HEIGHT)], fill=(15, 23, 42, 60), width=1)
    for gy in range(0, HEIGHT, 80):
        draw.line([(0, gy), (WIDTH, gy)], fill=(15, 23, 42, 60), width=1)
        
    # Glow ring
    draw.ellipse([WIDTH//2 - 420, HEIGHT//2 - 280, WIDTH//2 + 420, HEIGHT//2 + 280], outline=(14, 165, 233, 40), width=4)
    
    # Track badge
    badge_text = "MULTIMODAL AI HACKATHON 2026 • TRACK A"
    draw.rounded_rectangle([WIDTH//2 - 240, 240, WIDTH//2 + 240, 280], radius=10, fill=(30, 41, 59, 230), outline=(56, 189, 248, 120), width=1)
    draw.text((WIDTH//2 - 200, 248), badge_text, font=font_badge, fill=(56, 189, 248, 255))
    
    # Title
    main_title = "CARDIOVISION 3D"
    draw.text((WIDTH//2 - 260, 310), main_title, font=font_title, fill=(255, 255, 255, 255))
    
    # Subtitle
    subtitle = "Interactive AI Digital Twin for Multi-Vessel Coronary Perfusion & Occlusion Mapping"
    draw.text((WIDTH//2 - 510, 390), subtitle, font=font_subtitle, fill=(148, 163, 184, 255))
    
    # Author / Team
    team_text = "Presented by Team Singularity  •  Manas Jadhav, Prashant Kolekar, Shreya Jagtap, Advait Lad"
    draw.text((WIDTH//2 - 420, 480), team_text, font=font_body, fill=(203, 213, 225, 255))
    
    # Stack chips
    chips = [
        "Python 3.13 / FastAPI",
        "Three.js WebGL (r128)",
        "Scikit-Learn Random Forest",
        "SHAP Feature Attribution",
        "Dynamic Lead II ECG"
    ]
    cx_start = 280
    for chip in chips:
        draw.rounded_rectangle([cx_start, 570, cx_start + 240, 615], radius=8, fill=(15, 23, 42, 240), outline=(14, 165, 233, 160), width=1)
        draw.text((cx_start + 18, 582), chip, font=font_banner_sub, fill=(125, 211, 252, 255))
        cx_start += 270

    # System Status
    draw.text((WIDTH//2 - 190, 710), "● Live End-to-End System Demo", font=font_badge, fill=(52, 211, 153, 255))
    
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def create_summary_card():
    """Generates a polished closing summary card frame."""
    pil_img = Image.new("RGBA", (WIDTH, HEIGHT), (6, 9, 15, 255))
    draw = ImageDraw.Draw(pil_img, "RGBA")
    
    # Background Grid
    for gx in range(0, WIDTH, 80):
        draw.line([(gx, 0), (gx, HEIGHT)], fill=(15, 23, 42, 60), width=1)
    for gy in range(0, HEIGHT, 80):
        draw.line([(0, gy), (WIDTH, gy)], fill=(15, 23, 42, 60), width=1)

    draw.rounded_rectangle([WIDTH//2 - 220, 180, WIDTH//2 + 220, 220], radius=10, fill=(30, 41, 59, 230), outline=(52, 211, 153, 140), width=1)
    draw.text((WIDTH//2 - 180, 188), "HACKATHON DELIVERABLES READY", font=font_badge, fill=(52, 211, 153, 255))

    draw.text((WIDTH//2 - 270, 250), "CARDIOVISION 3D", font=font_title, fill=(255, 255, 255, 255))
    draw.text((WIDTH//2 - 340, 330), "Clinical Decision Support & Anatomical Digital Twin Prototype", font=font_subtitle, fill=(148, 163, 184, 255))

    features = [
        ("✔ Multi-Target Predictions", "Independent models for CAD (>50%), LAD, LCX, and RCA without mutual leakage"),
        ("✔ Zero Target Leakage", "Post-catheterization features eliminated; trained purely on pre-cath biomarkers"),
        ("✔ High Performance", "Achieves 0.931 ROC-AUC on the UCI Z-Alizadeh Sani Cardiovascular Dataset (n=303)"),
        ("✔ Continuous 3D Anatomy", "Catmull-Rom arterial splines overlaid onto 7.55MB ZBrush anatomical myocardium"),
        ("✔ Real-Time SHAP XAI", "Instant tree-level Shapley attribution ranking biomarkers driving individual risk"),
        ("✔ Open-Source Codebase", "github.com/prashantkolekar008/Cardiovascular-AI-project")
    ]

    y_pos = 420
    for title, desc in features:
        draw.text((WIDTH//2 - 440, y_pos), title, font=font_banner_h1, fill=(56, 189, 248, 255))
        draw.text((WIDTH//2 - 110, y_pos + 2), desc, font=font_banner_sub, fill=(203, 213, 225, 255))
        y_pos += 52

    draw.text((WIDTH//2 - 380, 800), "Team Singularity: Manas Jadhav, Prashant Kolekar, Shreya Jagtap, Advait Lad", font=font_body, fill=(148, 163, 184, 255))
    draw.text((WIDTH//2 - 230, 850), "Track A: Cardiovascular Risk Visualization & Prediction", font=font_badge, fill=(100, 116, 139, 255))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def main():
    print("Initializing video writer...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, float(FPS), (WIDTH, HEIGHT))

    if not out.isOpened():
        print("Error: Could not open VideoWriter.")
        return

    # Total timeline plan: ~52 seconds (1560 frames)
    # Scene 1: 0 - 120 (4s) Title Card
    print("Generating Scene 1: Title Card...")
    title_frame = create_title_card()
    for i in range(120):
        # Fade in first 20 frames
        if i < 20:
            alpha = i / 20.0
            f = cv2.addWeighted(title_frame, alpha, np.zeros_like(title_frame), 1.0 - alpha, 0)
        else:
            f = title_frame
        out.write(f)

    # Scene 2: 120 - 420 (10s) Case 03: Triple-Vessel CAD Overview & Zoom into 3D Heart
    print("Generating Scene 2: Case 03 High-Risk 3D Heart Inspection...")
    cursor_x, cursor_y = 1680, 50 # Case 03 button position
    for i in range(300):
        t = i / 300.0
        # Zoom from 1.0 to 1.5 into the 3D heart (center x=500, y=550)
        zoom = 1.0 + 0.65 * (1.0 - np.cos(t * np.pi)) / 2.0
        cx = 960 * (1 - t) + 530 * t
        cy = 540 * (1 - t) + 550 * t
        frame = crop_and_scale(img_case03, cx, cy, zoom)
        
        # Cursor action: click on Case 03 button in the first 45 frames
        click_p = (i - 15) / 25.0 if 15 <= i <= 40 else 0.0
        if i < 60:
            frame = draw_cursor(frame, cursor_x, cursor_y, click_p)
            
        frame = draw_lower_third(
            frame,
            title="PATIENT C: SEVERE TRIPLE-VESSEL CAD (91.3% PROBABILITY)",
            subtitle="Three.js 3D Anatomical Twin illuminates occluded LAD in red with pulsing lesion ring",
            tag="CASE 03: HIGH RISK"
        )
        out.write(frame)

    # Scene 3: 420 - 660 (8s) Pan to CAD-RADS Gauge & ST-Elevation Lead II ECG
    print("Generating Scene 3: CAD-RADS & Lead II ECG...")
    for i in range(240):
        t = i / 240.0
        # Pan across to CAD-RADS (x=1400, y=360) and then down to ECG (x=960, y=980)
        if t < 0.5:
            st = t * 2.0
            cx = 530 * (1 - st) + 1400 * st
            cy = 550 * (1 - st) + 360 * st
            zoom = 1.65 - 0.25 * st
            sub = "Ensemble models classify patient as CAD-RADS Category 4A (Severe Occlusion: 70-99%)"
        else:
            st = (t - 0.5) * 2.0
            cx = 1400 * (1 - st) + 600 * st
            cy = 360 * (1 - st) + 980 * st
            zoom = 1.4 + 0.3 * st
            sub = "Dynamic Lead II rhythm strip shows acute ST-segment elevation matching anterior ischemia"

        frame = crop_and_scale(img_case03, cx, cy, zoom)
        frame = draw_lower_third(
            frame,
            title="MULTI-TARGET CAD-RADS 4A & DYNAMIC LEAD II ECG",
            subtitle=sub,
            tag="HEMODYNAMIC DIAGNOSTICS"
        )
        out.write(frame)

    # Scene 4: 660 - 930 (9s) Explainable AI (SHAP Tree Attribution)
    print("Generating Scene 4: SHAP Explainability...")
    for i in range(270):
        t = i / 270.0
        # Zoom into SHAP panel (x=1500, y=820)
        cx = 600 * (1 - t) + 1480 * t
        cy = 980 * (1 - t) + 820 * t
        zoom = 1.7
        frame = crop_and_scale(img_case03, cx, cy, zoom)
        frame = draw_lower_third(
            frame,
            title="EXPLAINABLE AI: TREE-LEVEL SHAP ATTRIBUTION",
            subtitle="Rank-orders biomarkers: Typical Chest Pain (22.9%), LDL 215 mg/dL (15.3%), BP 165 mmHg (13.1%)",
            tag="CLINICAL INTERPRETABILITY"
        )
        out.write(frame)

    # Scene 5: 930 - 1200 (9s) Switching to Case 01: Low Risk
    print("Generating Scene 5: Recalibration to Low Risk Patient A...")
    for i in range(270):
        t = i / 270.0
        # Transition from Case 03 to Case 01
        if i < 30:
            # Cross dissolve
            alpha = i / 30.0
            frame = cv2.addWeighted(img_case01, alpha, img_case03, 1.0 - alpha, 0)
        else:
            # Zoom into Case 01 heart
            zoom = 1.0 + 0.55 * (1.0 - np.cos(t * np.pi)) / 2.0
            cx = 960 * (1 - t) + 520 * t
            cy = 540 * (1 - t) + 520 * t
            frame = crop_and_scale(img_case01, cx, cy, zoom)

        # Cursor clicking Case 01 button (x=1320, y=50) in early frames
        if i < 40:
            click_p = (i - 10) / 20.0 if 10 <= i <= 30 else 0.0
            frame = draw_cursor(frame, 1320, 50, click_p)

        frame = draw_lower_third(
            frame,
            title="DYNAMIC RECALIBRATION: HEALTHY PATIENT A (LOW RISK)",
            subtitle="Coronary tree shifts to all-green (<35% stenosis); ECG stabilizes to 68 BPM baseline sinus rhythm",
            tag="CASE 01: NORMAL PERFUSION"
        )
        out.write(frame)

    # Scene 6: 1200 - 1440 (8s) Clinical Parameters Simulation Modal
    print("Generating Scene 6: Vitals Simulation Modal...")
    for i in range(240):
        t = i / 240.0
        if i < 25:
            alpha = i / 25.0
            frame = cv2.addWeighted(img_modal, alpha, img_case01, 1.0 - alpha, 0)
        else:
            zoom = 1.0 + 0.2 * t
            cx = 960
            cy = 540
            frame = crop_and_scale(img_modal, cx, cy, zoom)

        # Cursor moving to 'Run Recalibration' button (x=1220, y=675)
        if 80 <= i <= 180:
            ct = (i - 80) / 100.0
            cur_x = 960 * (1 - ct) + 1220 * ct
            cur_y = 500 * (1 - ct) + 675 * ct
            click_p = (i - 150) / 20.0 if 150 <= i <= 170 else 0.0
            frame = draw_cursor(frame, cur_x, cur_y, click_p)

        frame = draw_lower_third(
            frame,
            title="IN-SILICO CLINICAL PARAMETERS SIMULATION",
            subtitle="Clinicians adjust hemodynamics & lipid biomarkers to stress-test real-time model recalibration",
            tag="INTERACTIVE SIMULATION"
        )
        out.write(frame)

    # Scene 7: 1440 - 1680 (8s) Closing Architecture Summary
    print("Generating Scene 7: Closing Summary & Specs...")
    summary_frame = create_summary_card()
    for i in range(240):
        if i < 25:
            alpha = i / 25.0
            frame = cv2.addWeighted(summary_frame, alpha, img_modal, 1.0 - alpha, 0)
        elif i > 215:
            alpha = (240 - i) / 25.0
            frame = cv2.addWeighted(summary_frame, alpha, np.zeros_like(summary_frame), 1.0 - alpha, 0)
        else:
            frame = summary_frame
        out.write(frame)

    out.release()
    print("Demo video generated successfully at:", OUTPUT_VIDEO_PATH)

    # Also copy to repo docs
    import shutil
    shutil.copyfile(OUTPUT_VIDEO_PATH, REPO_VIDEO_PATH)
    print("Copied to repo docs:", REPO_VIDEO_PATH)

if __name__ == "__main__":
    main()
