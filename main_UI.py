'''
# Project Configuration
This is the main file of our application.
To run this application, please follow these steps:

1. Python Version: 3.11.5
2. Install MySQL: Download MySQL
3. Register SQL account locally # For line 41 and 42
4. Install Dependencies: pip install -r requirements.txt
5. Modify the model.pt path in line 112
6. Run the Program

7. If the program doesn't create database automatically, 
you may run 'CREATE DATABASE IF NOT EXISTS classroom_db' in MySQL workbench


Demo Video:
Video 1: https://www.youtube.com/watch?v=RJjN9a0nDVM
Video 2: https://www.youtube.com/watch?v=ILIo-1Fy1-Y
'''


from ultralytics import YOLO
import cv2
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import time
import mysql.connector
from PIL import Image, ImageTk
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ======================== DATABASE SETUP ========================
def setup_database():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",   # Please change it to your user's name
            password="34870901", # # Please change it to your password
            database="classroom_db"
        )
        
        cursor = conn.cursor()
        cursor.execute("CREATE DATABASE IF NOT EXISTS classroom_db")
        cursor.execute("USE classroom_db")
        
        # Create separate tables for each class
        for class_name in ['6a', '6b']:
            cursor.execute(f'''CREATE TABLE IF NOT EXISTS incidents_{class_name}
                            (id INT AUTO_INCREMENT PRIMARY KEY,
                             student_id VARCHAR(10),
                             behavior VARCHAR(50),
                             timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                             duration INT DEFAULT 0)''')
        
        conn.commit()
        return conn
    except mysql.connector.Error as e:
        print(f"Database error: {e}")
        return None

# Mock student IDs
STUDENT_IDS = [f"STU-{i:03d}" for i in range(1, 31)]

# ======================== CORE LOGIC ========================
class BehaviorMonitor:
    """
    A real-time behavior monitoring system using YOLO object detection.
    
    Primary Objectives:
    1. Detect and track student behaviors (eating, sleeping, phone use, etc.)
    2. Generate alerts for specific behaviors with configurable thresholds
    3. Maintain persistent tracking of students across video frames
    4. Log incidents to a database for analysis
    
    Key Features:
    - Behavior detection using YOLO model
    - Persistent ID tracking across frames
    - Configurable alert thresholds
    - Visual annotation of detected behaviors
    - Database logging of incidents
    """
    def __init__(self, class_name):
        """
        Initialize the behavior monitoring system.
        
        Parameters:
        -----------
        class_name : str
            Identifier for the class/group being monitored (used for database tables)
            
        Initializes:
        ------------
        - YOLO model with pretrained weights
        - Behavior mapping between class IDs and names
        - Tracking and timing dictionaries
        - Database connection parameters
        """
        self.class_name = class_name.lower()
        self.sleep_trackers = {}
        self.current_behaviors = {}
        self.detection_active = False
        self.last_alert_time = {}
        self.last_detection_time = 0
        self.detection_interval = 1
        self.frame_count = 0
        self.track_id_map = {}  # For persistent ID tracking
        
        self.model_path = r"C:\Users\user\Desktop\Machine Learning\INT4097\Project\Code\model.pt"
        self.model = YOLO(self.model_path)
        print("Model loaded successfully. Classes:", self.model.names)
        
        self.behavior_map = {
            0: "Eating",
            1: "Looking_around",
            2: "Sleeping",
            3: "Watching_phone"
        }
        self.frame_count = 0
        self.start_time = time.time()

    def reset_statistics(self):
        """
        Reset all monitoring statistics and database records.
        
        Actions:
        --------
        1. Clears the incidents database table
        2. Resets all tracking dictionaries
        3. Maintains model state and configuration
        
        Raises:
        -------
        mysql.connector.Error
            If database operation fails
        """
        try:
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM incidents_{self.class_name}")
            conn.commit()
            self.sleep_trackers = {}
            self.current_behaviors = {}
            self.last_alert_time = {}
            print(f"Statistics reset for class {self.class_name}")
        except mysql.connector.Error as e:
            print(f"Error resetting statistics: {e}")

    def start_detection(self):
        self.detection_active = True
        self.sleep_trackers = {}
        self.current_behaviors = {}
        self.last_alert_time = {}

    def stop_detection(self):
        self.detection_active = False


    def _get_persisted_ids(self, count):
        """
        Generate or reuse persistent tracking IDs.
        
        Parameters:
        -----------
        count : int
            Number of IDs needed
            
        Returns:
        --------
        numpy.ndarray
            Array of persistent tracking IDs
            
        Implementation Notes:
        --------------------
        - Reuses existing IDs when possible to maintain continuity
        - Creates new sequential IDs when needed
        - Updates last-seen timestamp for each ID
        """
        # Generate persistent IDs using existing track IDs
        existing_ids = list(self.track_id_map.keys())
        new_ids = []
        
        for _ in range(count):
            if existing_ids:
                new_id = existing_ids.pop(0)
            else:
                new_id = max(existing_ids, default=0) + 1
            new_ids.append(new_id)
            self.track_id_map[new_id] = time.time()
            
        return np.array(new_ids) 

    def process_frame(self, frame):
        """
        Process a single video frame for behavior detection.
        
        Parameters:
        -----------
        frame : numpy.ndarray
            Input video frame in BGR format
            
        Returns:
        --------
        tuple
            (annotated_frame, alerts)
            - annotated_frame: Input frame with visual annotations
            - alerts: List of alert messages generated
            
        Processing Pipeline:
        -------------------
        1. Skip processing if detection inactive
        2. Run YOLO detection/tracking
        3. Annotate detected behaviors
        4. Trigger alerts based on behavior rules
        5. Return annotated frame and alerts
        """
        
        if not self.detection_active:
            return frame, []            


        alerts = []
        
        results = self.model.track(
            frame,
            persist=True,
            conf=0.5,
            tracker="botsort.yaml",  # or "bytetrack.yaml"
            verbose=False,
            imgsz=640,  # Match input size
            device='cpu'  # or '0' for GPU
        )
        
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            class_ids = r.boxes.cls.cpu().numpy().astype(int)
            track_ids = r.boxes.id.int().cpu().numpy() if r.boxes.id is not None else self._get_persisted_ids(len(boxes))

            for box, conf, cls_id, track_id in zip(boxes, confs, class_ids, track_ids):
                if conf > 0.5:
                    x1, y1, x2, y2 = map(int, box)
                    behavior = self.behavior_map.get(cls_id, "unknown")
                    student_id = f"STU-{track_id:03d}"
                    
                    self._draw_boxes(frame, x1, y1, x2, y2, behavior, student_id)
                    
                    if behavior == "Sleeping":
                        alert = self._handle_sleep_detection(student_id)
                        if alert:
                            alerts.append(alert)
                    else:
                        if student_id in self.sleep_trackers:
                            del self.sleep_trackers[student_id]
                        alert = self._trigger_alert(student_id, behavior)
                        if alert:
                            alerts.append(alert)

        return frame, alerts

    def _handle_sleep_detection(self, student_id):
        """
        Special handling for sleeping behavior detection.
        
        Parameters:
        -----------
        student_id : str
            Unique identifier for the student
            
        Returns:
        --------
        str or None
            Alert message if sleep duration threshold exceeded
            
        Logic Flow:
        ----------
        1. Track first detection time
        2. Calculate duration if already tracking
        3. Trigger alert after 5+ seconds
        4. Reset tracker after alert
        """
        current_time = time.time()
        if student_id not in self.sleep_trackers:
            self.sleep_trackers[student_id] = current_time
            return None
        
        sleep_duration = current_time - self.sleep_trackers[student_id]
        if sleep_duration >= 5:
            del self.sleep_trackers[student_id]  # Reset after alert
            return self._trigger_alert(student_id, "Sleeping", int(sleep_duration))
        return None

    def _draw_boxes(self, frame, x1, y1, x2, y2, behavior, student_id):
        colors = {
            "Sleeping": (50, 50, 255),    # Red in BGR
            "Eating": (0, 255, 0),        # Green in BGR
            "Looking_around": (0, 255, 255), # Yellow in BGR
            "Watching_phone": (0, 165, 255)  # Orange in BGR
        }
        color = colors.get(behavior, (0, 255, 0))
        
        # Create overlay and draw everything on it
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = f"{student_id}: {behavior.replace('_', ' ').title()}"
        cv2.putText(overlay, label, (x1, y1-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Blend overlay with original frame
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    def _trigger_alert(self, student_id, behavior, duration=0):
        """
        Generate and log behavior alerts.
        
        Parameters:
        -----------
        student_id : str
            Unique student identifier
        behavior : str
            Detected behavior class
        duration : int, optional
            Duration for timed behaviors (e.g., sleeping)
            
        Returns:
        --------
        tuple or None
            (alert_message, color_code) if alert should be raised
            None if alert was recently triggered
            
        Database Operations:
        -------------------
        - Inserts new incident record
        - Uses class-specific table
        - Includes timestamp, behavior and duration
        """
        current_time = time.time()
        colors = {
            "Sleeping": "#ff0000",    # Red
            "Eating": "#00ff00",      # Green
            "Looking_around": "#ffff00", # Yellow
            "Watching_phone": "#ffa500"  # Orange
        }
        color = colors.get(behavior, "#ffffff")
        alert_key = (student_id, behavior)
        if alert_key in self.last_alert_time and current_time - self.last_alert_time[alert_key] < 5:
            return None
        
        try:
            cursor = conn.cursor()
            cursor.execute(
                f"INSERT INTO incidents_{self.class_name} (student_id, behavior, duration) VALUES (%s, %s, %s)",
                (student_id, behavior, duration)
            )
            conn.commit()
            
            self.last_alert_time[alert_key] = current_time
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if behavior == "Sleeping":
                return (f"{timestamp} - {student_id}: {behavior} ({duration}s)", color)
            return (f"{timestamp} - {student_id}: {behavior}", color)
        except mysql.connector.Error as e:
            print(f"Error saving to database: {e}")
            return None

# ======================== START PAGE ========================
class StartPage:
    def __init__(self, root):
        self.root = root
        self.root.title("Classroom Monitor - Select Class")
        self.root.geometry("400x300")
        self.root.configure(bg="#2c3e50")
        
        global conn
        conn = setup_database()
        if not conn:
            messagebox.showerror("Error", "Failed to connect to database")
            return

        self.setup_ui()

    def setup_ui(self):
        tk.Label(self.root, text="Select Class", font=("Roboto", 20, "bold"),
                bg="#2c3e50", fg="white").pack(pady=30)
        
        btn_style = {"font": ("Roboto", 14), "width": 15, "pady": 10, "bg": "#3498db", "fg": "white"}
        
        tk.Button(self.root, text="Class 6A", command=lambda: self.start_monitor("6a"), **btn_style).pack(pady=20)
        tk.Button(self.root, text="Class 6B", command=lambda: self.start_monitor("6b"), **btn_style).pack(pady=20)

    def start_monitor(self, class_name):
        self.root.destroy()
        main_root = tk.Tk()
        app = ClassroomMonitorUI(main_root, class_name)
        main_root.protocol("WM_DELETE_WINDOW", app.on_closing)
        main_root.mainloop()

# ======================== MAIN UI ========================
class ClassroomMonitorUI:
    """
    A graphical user interface for real-time classroom behavior monitoring.
    
    Primary Objectives:
    1. Provide interactive visualization of live classroom monitoring
    2. Display real-time behavior alerts with color-coded severity
    3. Offer control interface for starting/stopping monitoring
    4. Present statistical analysis of detected behaviors
    
    Key Components:
    - Live video feed with behavior annotations
    - Alert logging system
    - Monitoring controls
    - Statistical visualization
    - Database integration
    """
    def __init__(self, root, class_name):
        """
        Initialize the monitoring interface.
        
        Parameters:
        -----------
        root : tk.Tk
            The main Tkinter window object
        class_name : str
            Name of the class being monitored (e.g., "6a")
            
        Initializes:
        ------------
        - Behavior monitoring backend
        - Video capture device
        - UI layout and styling
        - Alert tracking system
        """
        self.root = root
        self.class_name = class_name
        self.root.title(f"✨ AI Classroom Behavior Monitor - Class {class_name.upper()} ✨")
        self.root.geometry("1280x800")
        self.root.configure(bg="#2c3e50")
        
        self.monitor = BehaviorMonitor(class_name)
        self.cap = cv2.VideoCapture(0)
        self.is_monitoring = False
        self.last_alert_update = 0
        self.displayed_alerts = set()
        
        self.colors = {
            "primary": "#3498db",
            "secondary": "#2ecc71",
            "accent": "#e74c3c",
            "dark": "#2c3e50",
            "light": "#ecf0f1",
            "highlight": "#f1c40f"
        }
        
        self.setup_ui()
        self.update_frame()

    def setup_ui(self):
        """
        Configure the main application interface layout.
        
        Creates:
        --------
        1. Video display area
        2. Alert log panel
        3. Control buttons
        4. Status indicators
        
        Layout Structure:
        ----------------
        [ Video Frame ]  [ Alert Frame ]
        [         Control Frame        ]
        """
        self.root.grid_columnconfigure(0, weight=3)
        self.root.grid_columnconfigure(1, weight=2)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=0)
        
        self.video_frame = tk.Frame(self.root, bg=self.colors["dark"])
        self.video_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.video_canvas = tk.Canvas(self.video_frame, bg=self.colors["dark"])
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        
        self.alert_frame = tk.Frame(self.root, bg=self.colors["dark"])
        self.alert_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        alert_header = tk.Frame(self.alert_frame, bg=self.colors["primary"])
        alert_header.pack(fill=tk.X)
        tk.Label(alert_header, text="🚨 Behavior Alerts", font=("Roboto", 14, "bold"), 
                bg=self.colors["primary"], fg="white").pack(pady=5)
        
        self.alert_container = tk.Frame(self.alert_frame, bg=self.colors["dark"])
        self.alert_container.pack(fill=tk.BOTH, expand=True)
        
        self.scrollbar = ttk.Scrollbar(self.alert_container)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.alert_log = tk.Text(
            self.alert_container,
            height=25,
            width=50,
            bg="#34495e",
            fg="white",
            font=("Consolas", 10)
        )
        self.alert_log.tag_config('sleep', foreground="#ff0000")
        self.alert_log.tag_config('eat', foreground="#00ff00")
        self.alert_log.tag_config('look', foreground="#ffff00")
        self.alert_log.tag_config('phone', foreground="#ffa500")

        self.alert_log.pack(fill=tk.BOTH, expand=True)
        self.scrollbar.config(command=self.alert_log.yview)
        
        self.control_frame = tk.Frame(self.root, bg=self.colors["dark"])
        self.control_frame.grid(row=1, column=0, columnspan=2, pady=10, sticky="ew")
        
        btn_style = {"font": ("Roboto", 12), "borderwidth": 0, "width": 15, "padx": 20, "pady": 10}
        
        self.btn_start = tk.Button(self.control_frame, text="▶ Start", command=self.start_monitoring,
                                 bg=self.colors["secondary"], fg="white", **btn_style)
        self.btn_start.pack(side=tk.LEFT, padx=20)
        
        self.btn_stop = tk.Button(self.control_frame, text="⏹ Stop", command=self.stop_monitoring,
                                bg=self.colors["accent"], fg="white", state=tk.DISABLED, **btn_style)
        self.btn_stop.pack(side=tk.LEFT, padx=20)
        
        self.btn_backstage = tk.Button(self.control_frame, text="📋 Backstage", command=self.show_backstage,
                                     bg="#9b59b6", fg="white", **btn_style)
        self.btn_backstage.pack(side=tk.LEFT, padx=20)
        
        self.status_frame = tk.Frame(self.control_frame, bg=self.colors["dark"])
        self.status_frame.pack(side=tk.LEFT, padx=20)
        
        self.status_icon = tk.Label(self.status_frame, text="◉", font=("Arial", 12), 
                                  bg=self.colors["dark"], fg="gray")
        self.status_icon.pack(side=tk.LEFT)
        
        self.status_var = tk.StringVar(value="System Ready")
        self.status_label = tk.Label(self.status_frame, textvariable=self.status_var,
                                   font=("Roboto", 10, "bold"), bg=self.colors["dark"],
                                   fg=self.colors["light"])
        self.status_label.pack(side=tk.LEFT)
        
        self.btn_stats = tk.Button(self.control_frame, text="📊 Statistics", command=self.show_statistics,
                                 bg=self.colors["primary"], fg="white", **btn_style)
        self.btn_stats.pack(side=tk.RIGHT, padx=20)

        

        self.scrollbar.config(command=self.alert_log.yview)
        self.alert_log.config(yscrollcommand=self.scrollbar.set)

    def start_monitoring(self):
        """
        Activate the behavior monitoring system.
        
        Actions:
        --------
        1. Reset previous statistics
        2. Clear the alert log
        3. Enable detection in backend
        4. Update UI state
        """
        self.monitor.reset_statistics()
        # Change this line for Text widget clearing
        self.alert_log.delete('1.0', 'end')  # Changed from delete(0, tk.END)
        self.monitor.start_detection()
        self.is_monitoring = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.status_var.set("Monitoring Active")
        self.status_icon.config(fg="#2ecc71")

    def stop_monitoring(self):
        """
        Deactivate the behavior monitoring system.
        
        Actions:
        --------
        1. Stop detection in backend
        2. Clear the alert log
        3. Reset alert tracking
        4. Update UI state
        """
        self.monitor.stop_detection()
        self.alert_log.delete('1.0', 'end')
        self.displayed_alerts = set()  # Reset tracked alert


        self.is_monitoring = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.status_var.set("Monitoring Stopped")
        self.status_icon.config(fg="#e74c3c")

    def update_frame(self):
        """
        Continuous frame update loop for live video processing.
        
        Workflow:
        ---------
        1. Capture frame from video source
        2. Process frame for behavior detection
        3. Update alert log with new detections
        4. Display annotated video feed
        5. Schedule next update (30ms interval)
        """
        ret, frame = self.cap.read()
        if ret:
            processed_frame, alerts = self.monitor.process_frame(frame)
            
            # Process alerts in batches
            if alerts:
                for alert_data in alerts:
                    if alert_data:
                        alert_text, alert_color = alert_data
                        if alert_text not in self.displayed_alerts:
                            tag_name = alert_text.split(":")[1].strip().lower().replace(' ', '_')
                            self.alert_log.insert("end", alert_text + "\n", tag_name)
                            self.displayed_alerts.add(alert_text)
                self.alert_log.see("end")
            
            # Convert and display frame
            img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img).resize((800, 600), Image.LANCZOS)
            
            # Update display
            self.video_canvas.delete("all")
            self.imgtk = ImageTk.PhotoImage(image=img)
            self.video_canvas.create_image(
                self.video_canvas.winfo_width()//2,
                self.video_canvas.winfo_height()//2,
                anchor=tk.CENTER,
                image=self.imgtk
            )
        
        self.root.after(30, self.update_frame)

    

    def show_backstage(self):
        """
        Display the backstage statistics window.
        
        Features:
        --------
        - Class selection dropdown
        - Tabular behavior statistics
        - Manual refresh capability
        """
        backstage = tk.Toplevel(self.root)
        backstage.title("📋 Backstage Monitor")
        backstage.geometry("600x400")
        backstage.configure(bg=self.colors["dark"])
        
        tk.Label(backstage, text="Backstage Statistics", font=("Roboto", 16, "bold"),
                bg=self.colors["primary"], fg="white").pack(fill=tk.X, pady=(0,10))
        
        class_var = tk.StringVar(value="6a")
        tk.Label(backstage, text="Select Class:", bg=self.colors["dark"], fg="white").pack(pady=5)
        class_selector = ttk.Combobox(backstage, textvariable=class_var, values=["6a", "6b"], state="readonly")
        class_selector.pack(pady=5)
        
        stats_text = tk.Text(backstage, height=15, width=70, bg="#34495e", fg="white", font=("Consolas", 10))
        stats_text.pack(pady=10)
        
        def update_stats():
            try:
                cursor = conn.cursor()
                selected_class = class_var.get()
                cursor.execute(f"SELECT behavior, COUNT(*), SUM(duration) FROM incidents_{selected_class} GROUP BY behavior")
                results = cursor.fetchall()
                
                stats_text.delete(1.0, tk.END)
                stats_text.insert(tk.END, f"Statistics for Class {selected_class.upper()}\n\n")
                stats_text.insert(tk.END, "Behavior".ljust(20) + "Count".ljust(10) + "Total Duration\n")
                stats_text.insert(tk.END, "-"*50 + "\n")
                
                for behavior, count, duration in results:
                    duration_str = f"{duration}s" if behavior == "Sleeping" else "-"
                    stats_text.insert(tk.END, f"{behavior.ljust(20)}{str(count).ljust(10)}{duration_str}\n")
            except mysql.connector.Error as e:
                stats_text.delete(1.0, tk.END)
                stats_text.insert(tk.END, f"Error: {e}")
        
        tk.Button(backstage, text="Refresh Stats", command=update_stats,
                 bg=self.colors["primary"], fg="white").pack(pady=10)
        update_stats()

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.cap.release()
            if conn:
                conn.close()
            self.root.destroy()

    def show_statistics(self):
        """
        Display graphical behavior statistics.
        
        Features:
        --------
        - Interactive matplotlib chart
        - Comparison of behavior frequencies
        - Sleep duration visualization
        """
        try:
            cursor = conn.cursor()
            cursor.execute(f"SELECT behavior, COUNT(*), SUM(duration) FROM incidents_{self.class_name} GROUP BY behavior")
            results = cursor.fetchall()
            
            stats_window = tk.Toplevel(self.root)
            stats_window.title(f"📈 Detection Statistics - Class {self.class_name.upper()}")
            stats_window.geometry("800x600")
            stats_window.configure(bg=self.colors["dark"])

            if not results:
                tk.Label(stats_window, text="No detection data available", 
                        bg=self.colors["dark"], fg="white").pack(pady=20)
                return

            # Create figure for matplotlib
            fig = Figure(figsize=(8, 6), dpi=100)
            ax = fig.add_subplot(111)
            
            # Extract data for plotting
            behaviors = [row[0].replace('_', ' ').title() for row in results]
            counts = [row[1] for row in results]
            durations = [row[2] if row[0] == 'Sleeping' else 0 for row in results]

            # Create bars
            x = np.arange(len(behaviors))
            width = 0.35

            bars1 = ax.bar(x - width/2, counts, width, label='Incident Count', color='#3498db')
            bars2 = ax.bar(x + width/2, durations, width, label='Total Sleep Duration (s)', color='#e74c3c')

            # Add labels and styling
            ax.set_title(f'Behavior Statistics - Class {self.class_name.upper()}')
            ax.set_xticks(x)
            ax.set_xticklabels(behaviors, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_facecolor('#f5f6fa')

            # Add value labels on top of bars
            for bar in bars1:
                height = bar.get_height()
                ax.annotate(f'{height}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

            # Create canvas and add to window
            canvas = FigureCanvasTkAgg(fig, master=stats_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Add close button
            btn_frame = tk.Frame(stats_window, bg=self.colors["dark"])
            btn_frame.pack(pady=10)
            
            tk.Button(btn_frame, text="Close", command=stats_window.destroy,
                    bg=self.colors["accent"], fg="white").pack(side=tk.LEFT, padx=10)

        except mysql.connector.Error as e:
            messagebox.showerror("Error", f"Database error: {e}")

# ======================== RUN ========================
if __name__ == "__main__":
    root = tk.Tk()
    app = StartPage(root)
    root.mainloop()