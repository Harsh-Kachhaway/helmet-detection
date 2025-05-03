import cv2
import pytesseract
from ultralytics import YOLO
import datetime
import threading
import sqlite3
import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import csv
import json
import os

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


# Thread management improvements
threads = { }
flag_lock = threading.Lock()


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

    # Set a fixed window size (e.g., 640x480)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480)  # You can adjust the size as per your needs

    while True:
        with flag_lock:  # Lock the flag check to ensure thread safety
            if not running_flags.get(str(source), False):
                break

        ret, frame = cap.read()
        if not ret:
            if not cap.isOpened():
                print("🔌 VideoCapture not opened:", source)
            else:
                print("❌ Failed to grab frame from:", source)
            break

        # Process the frame with detection (your existing code)
        frame = process_frame(frame)

        # Resize frame to the set window size (optional)
        # frame = cv2.resize(frame, (640, 480))  # Resize the frame to fit the window size

        # Show the resized frame
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

        with flag_lock:
            if running_flags.get(source):
                continue  # Already running
            running_flags[source] = True

        t = threading.Thread(target=camera_thread, args=(source,), daemon=True)
        with flag_lock:
            threads[source] = t
        t.start()


def stop_detection(status_label):
    with flag_lock:
        for source in list(running_flags.keys()):
            running_flags[source] = False

    # Join threads to allow clean exit
    for source, t in list(threads.items()):
        if t.is_alive():
            t.join(timeout=2)
        with flag_lock:
            threads.pop(source, None)
            running_flags.pop(source, None)

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


URLS_FILE = "previous_urls.json"


def load_previous_urls():
    if os.path.exists(URLS_FILE):
        with open(URLS_FILE, 'r') as f:
            return json.load(f)
    return []


def save_previous_urls(previous_urls):
    with open(URLS_FILE, 'w') as f:
        json.dump(previous_urls, f)


def main():
    root = tk.Tk()
    root.title("Helmet & Number Plate Detection")

    # === Top Live + Date Labels ===
    top_frame = tk.Frame(root)
    top_frame.pack(fill='x', pady=5)

    date_label = tk.Label(top_frame, text=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), font=('Helvetica', 10))
    date_label.pack(side='left', padx=10)

    live_label_frame = tk.Frame(top_frame)
    live_label_frame.pack(side='right', padx=10)
    red_dot = tk.Label(live_label_frame, text="●", fg="gray", font=('Helvetica', 14, 'bold'))
    red_dot.pack(side='left')
    live_status_label = tk.Label(live_label_frame, text="Idle", fg='gray', font=('Helvetica', 10, 'bold'))
    live_status_label.pack(side='left')

    # === URL Input ===
    tk.Label(root, text="Enter Camera Indexes or Stream URLs (comma-separated):").pack(pady=5)
    urls_entry = tk.Entry(root, width=60)
    urls_entry.pack(pady=5)

    status_label = tk.Label(root, text="Idle")
    status_label.pack(pady=5)

    # === Start/Stop/View Buttons ===
    buttons_frame = tk.Frame(root)
    buttons_frame.pack(pady=5)

    # These will be linked later
    start_button = None
    stop_button = None

    # === Previously Used URLs Section ===
    prev_frame = tk.LabelFrame(root, text="Previously Used URLs")
    prev_frame.pack(pady=10, fill='both', padx=10)

    canvas = tk.Canvas(prev_frame, height=120)
    scrollbar = tk.Scrollbar(prev_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # === Reusable Logic ===
    previous_urls = load_previous_urls()

    def connect_single_url(url):
        urls_entry.delete(0, tk.END)
        urls_entry.insert(0, url)
        start_detection(urls_entry, status_label)

    def add_url_row(url):
        row = tk.Frame(scrollable_frame)
        row.pack(fill='x', pady=2, padx=5)

        label = tk.Label(row, text=url, anchor='w')
        label.pack(side='left', fill='x', expand=True)

        tk.Button(row, text="Connect", command=lambda u=url: connect_single_url(u)).pack(side='right', padx=5)

    for u in previous_urls:
        add_url_row(u)

    def start_and_store(entry_widget, status_label):
        input_text = entry_widget.get()
        urls = [u.strip() for u in input_text.split(',') if u.strip()]
        updated = False
        for u in urls:
            if u not in previous_urls:
                previous_urls.append(u)
                add_url_row(u)
                updated = True
        if updated:
            save_previous_urls(previous_urls)
        start_detection(entry_widget, status_label)

    def connect_all_urls():
        if not previous_urls:
            messagebox.showwarning("No URLs", "No previous URLs to connect.")
            return
        urls_entry.delete(0, tk.END)
        urls_entry.insert(0, ', '.join(previous_urls))
        start_detection(urls_entry, status_label)

    # === Buttons Finalize ===
    tk.Button(buttons_frame, text="Start Detection", command=lambda: start_and_store(urls_entry, status_label)).pack(
        side='left', padx=10)
    tk.Button(buttons_frame, text="Stop Detection", command=lambda: stop_detection(status_label)).pack(side='left',
                                                                                                       padx=10)
    tk.Button(root, text="View Detections", command=view_detections_window).pack(side='right', padx=10, pady=10)
    tk.Button(root, text="Connect All Previous URLs", command=connect_all_urls).pack(pady=5)

    # === Live Date & Status Updater ===
    def update_labels():
        date_label.config(text=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        if any(running_flags.values()):
            live_status_label.config(text="LIVE", fg="red")
            red_dot.config(fg="red")
        else:
            live_status_label.config(text="Idle", fg="gray")
            red_dot.config(fg="gray")
        root.after(1000, update_labels)

    update_labels()

    def on_close():
        save_previous_urls(previous_urls)
        quit_program(root, status_label)

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


def quit_program(root, status_label):
    stop_detection(status_label)
    cv2.destroyAllWindows()
    root.quit()
    root.destroy()


if __name__ == "__main__":
    main()
