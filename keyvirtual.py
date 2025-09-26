import cv2
import mediapipe as mp
import time

# ==== Inisialisasi MediaPipe ====
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# ==== Layout Keyboard ====
keys = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
    ["Z", "X", "C", "V", "B", "N", "M", "SPACE", "DEL", "ENTER"]
]

# ==== Ukuran tombol (diperkecil) ====
key_width = 40
key_height = 40
key_gap = 6
button_positions = []

# ==== Variabel input ====
typed_text = ""
last_pressed = None
press_start_time = 0
hold_time = 0.8  # detik untuk mendeteksi "tekan"

# ==== Fungsi untuk menulis teks dengan efek bold ====
def put_bold_text(img, text, position, font, scale, color, thickness):
    x, y = position
    # Shadow
    cv2.putText(img, text, (x+1, y+1), font, scale, (0, 0, 0), thickness + 2)
    # Text
    cv2.putText(img, text, (x, y), font, scale, color, thickness)

# ==== Gambar keyboard ====
def draw_keyboard(img, w, h, highlight_key=None):
    global button_positions
    button_positions = []

    bottom_margin = 100
    side_margin = 60
    offset_y = h - bottom_margin - (len(keys) * (key_height + key_gap)) + key_gap

    for row_index, row in enumerate(keys):
        total_row_width = len(row) * (key_width + key_gap) - key_gap
        offset_x = max((w - total_row_width) // 2, side_margin)

        if row_index == 1:
            offset_x += key_width // 2
        elif row_index == 2:
            offset_x += key_width

        for col_index, key in enumerate(row):
            x = offset_x + col_index * (key_width + key_gap)
            y = offset_y + row_index * (key_height + key_gap)

            button_positions.append((x, y, key))

            # Gambar kotak tombol
            if key == highlight_key:
                cv2.rectangle(img, (x, y), (x + key_width, y + key_height), (0, 255, 0), 2)
            else:
                cv2.rectangle(img, (x, y), (x + key_width, y + key_height), (255, 255, 255), 1)

            # Teks tombol
            label = key
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = x + (key_width - text_size[0]) // 2
            text_y = y + (key_height + text_size[1]) // 2
            put_bold_text(img, label, (text_x, text_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# ==== Cek apakah jari telunjuk berada di atas tombol ====
def check_key_press(x, y):
    for (bx, by, key) in button_positions:
        if bx < x < bx + key_width and by < y < by + key_height:
            return key
    return None

# ==== Tampilkan teks yang diketik ====
def draw_textbox(img, w):
    cv2.rectangle(img, (30, 20), (w - 30, 90), (255, 255, 255), -1)
    put_bold_text(img, typed_text, (50, 70),
                  cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

# ==== Main Program ====
cap = cv2.VideoCapture(0)
pTime = 0

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    draw_textbox(img, w)
    draw_keyboard(img, w, h)

    key_pressed = None
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            # Ambil koordinat ujung jari telunjuk (landmark 8)
            x_index = int(handLms.landmark[8].x * w)
            y_index = int(handLms.landmark[8].y * h)

            cv2.circle(img, (x_index, y_index), 8, (0, 255, 0), cv2.FILLED)

            # Jika ada lebih dari satu tangan, kita cek semua, tombol terakhir yang terdeteksi diutamakan
            current_key = check_key_press(x_index, y_index)
            if current_key is not None:
                key_pressed = current_key

        if key_pressed:
            if last_pressed != key_pressed:
                last_pressed = key_pressed
                press_start_time = time.time()
            else:
                if time.time() - press_start_time >= hold_time:
                    if key_pressed == "SPACE":
                        typed_text += " "
                    elif key_pressed == "DEL":
                        typed_text = typed_text[:-1]
                    elif key_pressed == "ENTER":
                        typed_text += "\n"
                    else:
                        typed_text += key_pressed
                    last_pressed = None
        else:
            last_pressed = None

        draw_keyboard(img, w, h, highlight_key=key_pressed)

    else:
        last_pressed = None

    # Tampilkan FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime) if pTime else 0
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Keyboard Virtual", img)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC untuk keluar
        break

cap.release()
cv2.destroyAllWindows()
