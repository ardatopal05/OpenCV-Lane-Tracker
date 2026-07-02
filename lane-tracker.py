import cv2
import numpy as np
import mss
import pydirectinput
import pygetwindow as gw
import time

pydirectinput.PAUSE = 0.00
WINDOW_TITLE = "Live for Speed" # Şu anda LFS autopilot gibi çalışıyor.
RESIZE_WIDTH = 320 
STEERING_THRESHOLD = 15

# KIRMIZI
LOWER_RED1, UPPER_RED1 = np.array([0, 60, 40]), np.array([15, 255, 255])
LOWER_RED2, UPPER_RED2 = np.array([160, 60, 40]), np.array([180, 255, 255])

# BEYAZ
# Saturation (Doygunluk) 0-60 arası (Rengi olmayanlar)
# Value (Parlaklık) 130-255 arası (Parlak olanlar)
# Eğer asfaltı yol sanarsa, 130 değerini(ilk np.array'in 3. elemanı) 150 veya 160'a çıkart.
LOWER_WHITE, UPPER_WHITE = np.array([0, 0, 130]), np.array([180, 60, 255])

def get_scan_area():
    try:
        win = gw.getWindowsWithTitle(WINDOW_TITLE)[0]
        x, y, w, h = win.left, win.top, win.width, win.height
        return {
            "top": y + int(h * 0.45), 
            "left": x + 10, 
            "width": w - 20, 
            "height": int(h * 0.25) 
        }
    except IndexError:
        return None

def apply_clahe(frame):
    # Karanlık bölgeleri aydınlatma(gölge v.b. için)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def process_hybrid_hull(frame):
    h, w, _ = frame.shape
    scale = RESIZE_WIDTH / w
    frame_small = cv2.resize(frame, (RESIZE_WIDTH, int(h * scale)))
    
    # Clahe
    frame_lit = apply_clahe(frame_small)
    h_small, w_small, _ = frame_lit.shape
    
    # Mask
    hsv = cv2.cvtColor(frame_lit, cv2.COLOR_BGR2HSV)
    
    # Red Mask
    mask_r1 = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1)
    mask_r2 = cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
    mask_red = cv2.bitwise_or(mask_r1, mask_r2)
    
    # White Maske
    mask_white = cv2.inRange(hsv, LOWER_WHITE, UPPER_WHITE)
    
    # Combine
    mask_combined = cv2.bitwise_or(mask_red, mask_white)
    
    # Parlamaları azaltmak/önlemek için erosion işlemi
    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel)
    
    # Convex Hull
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    screen_center_x = w_small // 2
    left_points = []
    right_points = []
    
    hull_mask = np.zeros_like(mask_clean)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 50: 
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                if cx < screen_center_x:
                    left_points.append(cnt)
                else:
                    right_points.append(cnt)

    left_limit = 0
    right_limit = w_small
    
    # Sol duvar
    if left_points:
        all_left = np.vstack(left_points)
        hull_left = cv2.convexHull(all_left)
        cv2.drawContours(hull_mask, [hull_left], -1, 255, -1)
        # En sa� u� (Yola en yak�n)
        left_limit = tuple(hull_left[hull_left[:, :, 0].argmax()][0])[0]

    # Sağ duvar
    if right_points:
        all_right = np.vstack(right_points)
        hull_right = cv2.convexHull(all_right)
        cv2.drawContours(hull_mask, [hull_right], -1, 255, -1)
        # En sol u� (Yola en yak�n)
        right_limit = tuple(hull_right[hull_right[:, :, 0].argmin()][0])[0]

    # Ekranın ortasını belirleme
    if right_limit <= left_limit:
        target_x = screen_center_x
        color = (0, 0, 255)
    else:
        target_x = int((left_limit + right_limit) / 2)
        color = (0, 255, 0)
        
    error = target_x - screen_center_x
    
    # Çizim
    cv2.line(frame_lit, (left_limit, 0), (left_limit, h_small), (255, 0, 0), 2)
    cv2.line(frame_lit, (right_limit, 0), (right_limit, h_small), (255, 0, 0), 2)
    cv2.line(frame_lit, (target_x, 0), (target_x, h_small), color, 2)
    cv2.circle(frame_lit, (screen_center_x, h_small//2), 4, (255, 255, 255), -1)

    return frame_lit, error, hull_mask

# Ana döngü
scan_region = get_scan_area()

if scan_region:
    with mss.mss() as sct:
        print("Hibrit Mod (Kirmizi + Beyaz + Golge) Aktif!")
        
        while True:
            t_start = time.time()
            img = np.array(sct.grab(scan_region))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            processed_frame, error, mask_debug = process_hybrid_hull(frame)
            
            if error > STEERING_THRESHOLD:
                pydirectinput.keyDown('d')
                pydirectinput.keyUp('a')
                cv2.putText(processed_frame, f"SAG > {error}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            elif error < -STEERING_THRESHOLD:
                pydirectinput.keyDown('a')
                pydirectinput.keyUp('d')
                cv2.putText(processed_frame, f"< SOL {error}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                pydirectinput.keyUp('a')
                pydirectinput.keyUp('d')
                cv2.putText(processed_frame, f"^ DUZ {error}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            fps = 1 / (time.time() - t_start)
            cv2.putText(processed_frame, f"FPS: {int(fps)}", (processed_frame.shape[1]-60, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow("Robot Gozu (Aydinlatilmis)", processed_frame)
            cv2.imshow("Maske (Solid Walls)", mask_debug)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
else:
    print("LFS Penceresi bulunamadi!")