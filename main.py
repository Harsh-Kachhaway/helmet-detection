import cv2
import pytesseract
from ultralytics import YOLO
import datetime
import threading
import sqlite3
import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import csv

# Load models
numberplate_model = YOLO("models/yolo11_numberplate.pt")
bike_model = YOLO("models/yolo11_bikedetection.pt")
helmet_model = YOLO("models/yolo11_helmetdetection.pt")

# Threading control
running_flags = {}
ocr_data = []
data_lock = threading.Lock()

# SQLite setup
conn = sqlite3.connect("detection_data.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        plate_text TEXT,
        confidence REAL
    )
''')
conn.commit()

def save_to_db(data):
    with data_lock:
        cursor.execute("INSERT INTO detections (timestamp, plate_text, confidence) VALUES (?, ?, ?)",
                       (data['Timestamp'], data['Plate Text'], data['Confidence']))
        conn.commit()

def process_frame(frame):
    timestamp_text = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Remove the "LIVE" overlay from the frame

    results_plate = numberplate_model(frame)
    for result in results_plate:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            plate_crop = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            plate_text = pytesseract.image_to_string(gray, config='--psm 7').strip()

            if plate_text and conf >= 0.75:
                data = {
                    "Timestamp": timestamp_text,
                    "Plate Text": plate_text,
                    "Confidence": round(conf, 2)
                }
                save_to_db(data)
                cv2.putText(frame, plate_text, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)

    results_bike = bike_model(frame)
    bikes = []
    for result in results_bike:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bikes.append((x1, y1, x2, y2))
            label = result.names[int(box.cls[0])]
            conf = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'{label} ({conf:.2f})', (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    results_helmet = helmet_model(frame)
    for result in results_helmet:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            inside_bike = any(bx1 < x1 < bx2 and by1 < y1 < by2 for bx1, by1, bx2, by2 in bikes)
            color = (0, 255, 0) if inside_bike else (0, 0, 255)
            tag = "Helmet" if inside_bike else "No Helmet"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{tag} ({conf:.2f})', (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame

def camera_thread(source):
    try:
        source = int(source)
    except ValueError:
        if source.startswith("http") and not source.endswith("/video"):
            source += "/video"

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open source {source}.")
        return

    print(f"Stream {source} started.")
    window_name = f"Live Detection - {source}"

    while running_flags.get(str(source), False):
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame)
        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(window_name)
    print(f"Stream {source} stopped.")

def start_detection(urls_entry, status_label):
    inputs = urls_entry.get().split(',')
    if not inputs:
        messagebox.showwarning("Input Error", "Please enter at least one camera index or URL")
        return

    status_label.config(text="Detection running...")
    for source in inputs:
        source = source.strip()
        if not source:
            continue

        running_flags[source] = True
        t = threading.Thread(target=camera_thread, args=(source,))
        t.start()

def stop_detection(status_label):
    for source in list(running_flags.keys()):
        running_flags[source] = False
    status_label.config(text="Stopped")

def export_to_csv():
    filename = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("CSV files", "*.csv")])
    if not filename:
        return

    cursor.execute("SELECT * FROM detections ORDER BY id DESC")
    rows = cursor.fetchall()

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "Timestamp", "Plate Text", "Confidence"])
        for row in rows:
            writer.writerow(row)

def view_detections_window():
    db_window = tk.Toplevel()
    db_window.title("Detection Records")

    search_frame = tk.Frame(db_window)
    search_frame.pack(fill='x')

    search_entry = tk.Entry(search_frame)
    search_entry.pack(side='left', fill='x', expand=True, padx=5, pady=5)

    def filter_data():
        query = search_entry.get()
        for i in tree.get_children():
            tree.delete(i)
        cursor.execute("SELECT * FROM detections WHERE plate_text LIKE ? ORDER BY id DESC", (f"%{query}%",))
        for row in cursor.fetchall():
            tree.insert("", "end", values=row)

    tk.Button(search_frame, text="Search", command=filter_data).pack(side='left', padx=5)
    tk.Button(search_frame, text="Export CSV", command=export_to_csv).pack(side='right', padx=5)

    tree = ttk.Treeview(db_window, columns=("ID", "Timestamp", "Plate Text", "Confidence"), show='headings')
    tree.heading("ID", text="ID")
    tree.heading("Timestamp", text="Timestamp")
    tree.heading("Plate Text", text="Plate Text")
    tree.heading("Confidence", text="Confidence")

    scrollbar = ttk.Scrollbar(db_window, orient="vertical", command=tree.yview)
    tree.configure(yscroll=scrollbar.set)
    scrollbar.pack(side='right', fill='y')
    tree.pack(fill='both', expand=True)

    cursor.execute("SELECT * FROM detections ORDER BY id DESC")
    for row in cursor.fetchall():
        tree.insert("", "end", values=row)
def main():
    root = tk.Tk()
    root.title("Helmet & Number Plate Detection")

    # Frame for date and live status labels
    top_frame = tk.Frame(root)
    top_frame.pack(fill='x', pady=5)

    # Date label on the left side (shows Date and Time)
    date_label = tk.Label(top_frame, text=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), font=('Helvetica', 10))
    date_label.pack(side='left', padx=10)

    # Live label on the right side with red dot when LIVE
    live_label_frame = tk.Frame(top_frame)
    live_label_frame.pack(side='right', padx=10)

    # Red dot label
    red_dot = tk.Label(live_label_frame, text="●", fg="red", font=('Helvetica', 14, 'bold'))
    red_dot.pack(side='left')

    # LIVE text label
    live_status_label = tk.Label(live_label_frame, text="LIVE", fg='red', font=('Helvetica', 10, 'bold'))
    live_status_label.pack(side='left')

    # Entry for camera URLs
    tk.Label(root, text="Enter Camera Indexes or Stream URLs (comma-separated):").pack(pady=5)
    urls_entry = tk.Entry(root, width=60)
    urls_entry.pack(pady=5)

    status_label = tk.Label(root, text="Idle")
    status_label.pack(pady=5)

    buttons_frame = tk.Frame(root)
    buttons_frame.pack(pady=5)

    tk.Button(buttons_frame, text="Start Detection", command=lambda: start_detection(urls_entry, status_label)).pack(side='left', padx=10)
    tk.Button(buttons_frame, text="Stop Detection", command=lambda: stop_detection(status_label)).pack(side='left', padx=10)

    tk.Button(root, text="View Detections", command=view_detections_window).pack(side='right', padx=10, pady=10)

    # Refresh the date and live status dynamically
    def update_labels():
        # Update date and time
        date_label.config(text=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # Update live status and red dot visibility
        if any(running_flags.values()):
            live_status_label.config(text="LIVE")
            red_dot.config(fg="red")
        else:
            live_status_label.config(text="Idle")
            red_dot.config(fg="gray")

        # Refresh every second
        root.after(1000, update_labels)

    update_labels()

    root.protocol("WM_DELETE_WINDOW", lambda: quit_program(root, status_label))
    root.mainloop()

def quit_program(root, status_label):
    stop_detection(status_label)
    cv2.destroyAllWindows()
    root.quit()
    root.destroy()

if __name__ == "__main__":
    main()
