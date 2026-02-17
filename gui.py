import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import threading
from ultralytics import YOLO
import os
from datetime import datetime
import time
import queue
import numpy as np


class ModernButton(tk.Canvas):
    def __init__(self, parent, text, command, bg_color="#3b82f6", hover_color="#2563eb", 
                 fg_color="white", width=200, height=45, radius=10, font_size=11, **kwargs):
        super().__init__(parent, width=width, height=height, bg=parent["bg"], 
                        highlightthickness=0, **kwargs)
        
        self.command = command
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.fg_color = fg_color
        self.current_color = bg_color
        self.text = text
        self.radius = radius
        self.font_size = font_size
        self._width = width
        self._height = height
        self.enabled = True
        
        self.draw_button()
        
        self.bind("<Enter>", self.on_hover)
        self.bind("<Leave>", self.on_leave)
        self.bind("<Button-1>", self.on_click)
        
    def draw_button(self):
        self.delete("all")
        self.create_rounded_rect(2, 2, self._width-2, self._height-2, 
                                self.radius, fill=self.current_color, outline="")
        self.create_text(self._width//2, self._height//2, text=self.text, 
                        fill=self.fg_color, font=("Segoe UI", self.font_size, "bold"))
        
    def create_rounded_rect(self, x1, y1, x2, y2, radius, **kwargs):
        points = [
            x1+radius, y1, x2-radius, y1, x2, y1, x2, y1+radius,
            x2, y2-radius, x2, y2, x2-radius, y2, x1+radius, y2,
            x1, y2, x1, y2-radius, x1, y1+radius, x1, y1
        ]
        return self.create_polygon(points, smooth=True, **kwargs)
        
    def on_hover(self, event):
        if self.enabled:
            self.current_color = self.hover_color
            self.draw_button()
        
    def on_leave(self, event):
        if self.enabled:
            self.current_color = self.bg_color
            self.draw_button()
        
    def on_click(self, event):
        if self.enabled and self.command:
            self.command()
            
    def set_state(self, enabled):
        self.enabled = enabled
        if not enabled:
            self.current_color = "#4b5563"
        else:
            self.current_color = self.bg_color
        self.draw_button()


class HelmetDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Helmet Detection System")
        self.root.geometry("1400x850")
        self.root.minsize(1200, 700)
        
        self.colors = {
            'bg_dark': '#0f0f0f',
            'bg_card': '#1a1a1a',
            'bg_card_hover': '#252525',
            'accent_blue': '#3b82f6',
            'accent_green': '#10b981',
            'accent_red': '#ef4444',
            'accent_orange': '#f59e0b',
            'accent_purple': '#8b5cf6',
            'text_primary': '#ffffff',
            'text_secondary': '#9ca3af',
            'border': '#2d2d2d'
        }
        
        self.root.configure(bg=self.colors['bg_dark'])
        
        # Load model
        self.model = None
        self.load_model()
        
        # Variables
        self.is_webcam_running = False
        self.is_video_running = False
        self.cap = None
        self.video_thread = None
        
        # Statistics
        self.total_violations = 0
        self.total_safe = 0
        self.frame_count = 0
        
        self.frame_skip = 1  
        self.process_skip = 3  
        self.last_update_time = time.time()
        self.fps = 0
        self.last_detections = [] 
        self.frame_queue = queue.Queue(maxsize=3)
        
        # Target resolution for processing (lower = faster)
        self.process_width = 416  
        self.process_height = 416
        
        # Display resolution
        self.display_width = 854
        self.display_height = 480
        
        # Setup GUI
        self.setup_gui()
        
        # Start display update loop
        self.update_display_loop()
        
        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def load_model(self):
        try:
            model_path = 'best.pt'
            if not os.path.exists(model_path):
                model_path = '/app/best.pt'
            
            self.model = YOLO(model_path)
            
            print("Model loaded successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}\nMake sure 'best.pt' is in the same directory.")
            
    def setup_gui(self):
        # Configure grid
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        # Left Panel - Controls
        self.setup_left_panel()
        
        # Right Panel - Video Display
        self.setup_right_panel()
        
    def setup_left_panel(self):
        """Setup left control panel"""
        
        left_panel = tk.Frame(self.root, bg=self.colors['bg_card'], width=300)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(15, 8), pady=15)
        left_panel.grid_propagate(False)
        
        title_frame = tk.Frame(left_panel, bg=self.colors['bg_card'])
        title_frame.pack(fill=tk.X, padx=20, pady=(25, 5))
        
        title_label = tk.Label(
            title_frame,
            text="HELMET DETECT",
            font=('Segoe UI', 18, 'bold'),
            bg=self.colors['bg_card'],
            fg=self.colors['text_primary']
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            title_frame,
            text="Safety Monitor",
            font=('Segoe UI', 10),
            bg=self.colors['bg_card'],
            fg=self.colors['text_secondary']
        )
        subtitle_label.pack()
        
        # Separator
        sep = tk.Frame(left_panel, bg=self.colors['border'], height=1)
        sep.pack(fill=tk.X, padx=20, pady=20)
        
        # Control Buttons Section
        controls_label = tk.Label(
            left_panel,
            text="DETECTION MODES",
            font=('Segoe UI', 9, 'bold'),
            bg=self.colors['bg_card'],
            fg=self.colors['text_secondary']
        )
        controls_label.pack(anchor='w', padx=25, pady=(0, 10))
        
        # Button container
        btn_container = tk.Frame(left_panel, bg=self.colors['bg_card'])
        btn_container.pack(fill=tk.X, padx=20)
        
        # Image Button
        self.btn_image = ModernButton(
            btn_container,
            text="📷  Detect Image",
            command=self.detect_image,
            bg_color=self.colors['accent_blue'],
            hover_color="#2563eb",
            width=260,
            height=48
        )
        self.btn_image.pack(pady=5)
        
        # Video Button
        self.btn_video = ModernButton(
            btn_container,
            text="🎬  Process Video",
            command=self.detect_video,
            bg_color=self.colors['accent_purple'],
            hover_color="#7c3aed",
            width=260,
            height=48
        )
        self.btn_video.pack(pady=5)
        
        # Webcam Buttons Frame
        webcam_frame = tk.Frame(btn_container, bg=self.colors['bg_card'])
        webcam_frame.pack(fill=tk.X, pady=5)
        
        self.btn_webcam_start = ModernButton(
            webcam_frame,
            text="▶ Webcam",
            command=self.start_webcam,
            bg_color=self.colors['accent_green'],
            hover_color="#059669",
            width=125,
            height=48
        )
        self.btn_webcam_start.pack(side=tk.LEFT, padx=(0, 5))
        
        self.btn_webcam_stop = ModernButton(
            webcam_frame,
            text="⏹ Stop",
            command=self.stop_webcam,
            bg_color=self.colors['accent_red'],
            hover_color="#dc2626",
            width=125,
            height=48
        )
        self.btn_webcam_stop.pack(side=tk.LEFT)
        self.btn_webcam_stop.set_state(False)
        
        # Separator
        sep2 = tk.Frame(left_panel, bg=self.colors['border'], height=1)
        sep2.pack(fill=tk.X, padx=20, pady=20)
        
        # Statistics Section
        stats_label = tk.Label(
            left_panel,
            text="LIVE STATISTICS",
            font=('Segoe UI', 9, 'bold'),
            bg=self.colors['bg_card'],
            fg=self.colors['text_secondary']
        )
        stats_label.pack(anchor='w', padx=25, pady=(0, 15))
        
        stats_container = tk.Frame(left_panel, bg=self.colors['bg_card'])
        stats_container.pack(fill=tk.X, padx=20)
        
        # Stats Grid
        self.setup_stat_card(stats_container, "FPS", "fps", self.colors['accent_green'], 0, 0)
        self.setup_stat_card(stats_container, "Frames", "frames", self.colors['accent_blue'], 0, 1)
        self.setup_stat_card(stats_container, "Violations", "violations", self.colors['accent_red'], 1, 0)
        self.setup_stat_card(stats_container, "Safe", "safe", self.colors['accent_green'], 1, 1)
        
        # Separator
        sep3 = tk.Frame(left_panel, bg=self.colors['border'], height=1)
        sep3.pack(fill=tk.X, padx=20, pady=20)
        
        # Settings Section
        settings_label = tk.Label(
            left_panel,
            text="SETTINGS",
            font=('Segoe UI', 9, 'bold'),
            bg=self.colors['bg_card'],
            fg=self.colors['text_secondary']
        )
        settings_label.pack(anchor='w', padx=25, pady=(0, 10))
        
        settings_container = tk.Frame(left_panel, bg=self.colors['bg_card'])
        settings_container.pack(fill=tk.X, padx=20)
        
        # Confidence Slider
        conf_frame = tk.Frame(settings_container, bg=self.colors['bg_card'])
        conf_frame.pack(fill=tk.X, pady=5)
        
        conf_header = tk.Frame(conf_frame, bg=self.colors['bg_card'])
        conf_header.pack(fill=tk.X)
        
        tk.Label(
            conf_header,
            text="Confidence",
            font=('Segoe UI', 10),
            bg=self.colors['bg_card'],
            fg=self.colors['text_primary']
        ).pack(side=tk.LEFT)
        
        self.lbl_conf_value = tk.Label(
            conf_header,
            text="50%",
            font=('Segoe UI', 10, 'bold'),
            bg=self.colors['bg_card'],
            fg=self.colors['accent_blue']
        )
        self.lbl_conf_value.pack(side=tk.RIGHT)
        
        self.confidence_var = tk.DoubleVar(value=0.5)
        
        style = ttk.Style()
        style.configure("Custom.Horizontal.TScale", background=self.colors['bg_card'])
        
        self.confidence_scale = ttk.Scale(
            conf_frame,
            from_=0.1,
            to=0.9,
            variable=self.confidence_var,
            orient=tk.HORIZONTAL,
            style="Custom.Horizontal.TScale",
            command=self.update_confidence_label
        )
        self.confidence_scale.pack(fill=tk.X, pady=5)
        
        # Performance Mode
        perf_frame = tk.Frame(settings_container, bg=self.colors['bg_card'])
        perf_frame.pack(fill=tk.X, pady=10)
        
        perf_header = tk.Frame(perf_frame, bg=self.colors['bg_card'])
        perf_header.pack(fill=tk.X)
        
        tk.Label(
            perf_header,
            text="Performance Mode",
            font=('Segoe UI', 10),
            bg=self.colors['bg_card'],
            fg=self.colors['text_primary']
        ).pack(side=tk.LEFT)
        
        self.lbl_perf_value = tk.Label(
            perf_header,
            text="Balanced",
            font=('Segoe UI', 10, 'bold'),
            bg=self.colors['bg_card'],
            fg=self.colors['accent_orange']
        )
        self.lbl_perf_value.pack(side=tk.RIGHT)
        
        self.perf_var = tk.IntVar(value=2)
        self.perf_scale = ttk.Scale(
            perf_frame,
            from_=1,
            to=3,
            variable=self.perf_var,
            orient=tk.HORIZONTAL,
            style="Custom.Horizontal.TScale",
            command=self.update_perf_mode
        )
        self.perf_scale.pack(fill=tk.X, pady=5)
        
        perf_labels = tk.Frame(perf_frame, bg=self.colors['bg_card'])
        perf_labels.pack(fill=tk.X)
        tk.Label(perf_labels, text="Quality", font=('Segoe UI', 8), 
                bg=self.colors['bg_card'], fg=self.colors['text_secondary']).pack(side=tk.LEFT)
        tk.Label(perf_labels, text="Speed", font=('Segoe UI', 8), 
                bg=self.colors['bg_card'], fg=self.colors['text_secondary']).pack(side=tk.RIGHT)
        
        # Reset Button
        reset_container = tk.Frame(left_panel, bg=self.colors['bg_card'])
        reset_container.pack(fill=tk.X, padx=20, pady=15)
        
        self.btn_reset = ModernButton(
            reset_container,
            text="🔄  Reset Statistics",
            command=self.reset_stats,
            bg_color=self.colors['accent_orange'],
            hover_color="#d97706",
            width=260,
            height=42,
            font_size=10
        )
        self.btn_reset.pack()
        
    def setup_stat_card(self, parent, label, key, color, row, col):
        """Create a stat card widget"""
        frame = tk.Frame(parent, bg=self.colors['bg_dark'], padx=12, pady=10)
        frame.grid(row=row, column=col, padx=3, pady=3, sticky="nsew")
        parent.grid_columnconfigure(col, weight=1)
        
        value_label = tk.Label(
            frame,
            text="0",
            font=('Segoe UI', 22, 'bold'),
            bg=self.colors['bg_dark'],
            fg=color
        )
        value_label.pack()
        
        name_label = tk.Label(
            frame,
            text=label,
            font=('Segoe UI', 9),
            bg=self.colors['bg_dark'],
            fg=self.colors['text_secondary']
        )
        name_label.pack()
        
        # Store reference
        setattr(self, f'stat_{key}', value_label)
        
    def setup_right_panel(self):
        """Setup right video display panel"""
        
        right_panel = tk.Frame(self.root, bg=self.colors['bg_dark'])
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(8, 15), pady=15)
        
        # Video Container with border effect
        video_outer = tk.Frame(right_panel, bg=self.colors['border'], padx=2, pady=2)
        video_outer.pack(fill=tk.BOTH, expand=True)
        
        video_container = tk.Frame(video_outer, bg=self.colors['bg_card'])
        video_container.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = tk.Frame(video_container, bg=self.colors['bg_card'])
        header_frame.pack(fill=tk.X, padx=20, pady=(15, 10))
        
        # Live indicator
        live_frame = tk.Frame(header_frame, bg=self.colors['bg_card'])
        live_frame.pack(side=tk.LEFT)
        
        self.live_dot = tk.Label(
            live_frame,
            text="●",
            font=('Segoe UI', 12),
            bg=self.colors['bg_card'],
            fg=self.colors['text_secondary']
        )
        self.live_dot.pack(side=tk.LEFT)
        
        self.live_label = tk.Label(
            live_frame,
            text=" READY",
            font=('Segoe UI', 11, 'bold'),
            bg=self.colors['bg_card'],
            fg=self.colors['text_secondary']
        )
        self.live_label.pack(side=tk.LEFT)
        
        # Status
        self.status_label = tk.Label(
            header_frame,
            text="Select a detection mode to begin",
            font=('Segoe UI', 10),
            bg=self.colors['bg_card'],
            fg=self.colors['text_secondary']
        )
        self.status_label.pack(side=tk.RIGHT)
        
        # Video Canvas
        canvas_frame = tk.Frame(video_container, bg='#000000')
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        self.canvas = tk.Canvas(
            canvas_frame,
            bg='#000000',
            highlightthickness=0
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Draw placeholder
        self.draw_placeholder()
        
    def draw_placeholder(self):
        """Draw placeholder on canvas"""
        self.root.update_idletasks()
        w = self.canvas.winfo_width() or 800
        h = self.canvas.winfo_height() or 600
        
        self.canvas.delete("all")
        
        # Center text
        self.canvas.create_text(
            w // 2, h // 2 - 20,
            text="⛑",
            font=('Segoe UI Emoji', 48),
            fill='#333333'
        )
        self.canvas.create_text(
            w // 2, h // 2 + 40,
            text="No Active Feed",
            font=('Segoe UI', 14),
            fill='#444444'
        )
        self.canvas.create_text(
            w // 2, h // 2 + 65,
            text="Select Image, Video, or Start Webcam",
            font=('Segoe UI', 10),
            fill='#333333'
        )
        
    def update_confidence_label(self, value):
        """Update confidence threshold label"""
        val = float(value)
        self.lbl_conf_value.config(text=f"{int(val*100)}%")
        
    def update_perf_mode(self, value):
        """Update performance mode"""
        mode = int(float(value))
        if mode == 1:
            self.process_skip = 1  # Process every frame
            self.process_width = 640
            self.process_height = 640
            self.lbl_perf_value.config(text="Quality", fg=self.colors['accent_blue'])
        elif mode == 2:
            self.process_skip = 2  # Process every 2nd frame
            self.process_width = 416
            self.process_height = 416
            self.lbl_perf_value.config(text="Balanced", fg=self.colors['accent_orange'])
        else:
            self.process_skip = 3  # Process every 3rd frame
            self.process_width = 320
            self.process_height = 320
            self.lbl_perf_value.config(text="Speed", fg=self.colors['accent_green'])
        
    def reset_stats(self):
        """Reset statistics"""
        self.total_violations = 0
        self.total_safe = 0
        self.frame_count = 0
        self.fps = 0
        self.update_stats()
        self.status_label.config(text="Statistics reset")
        
    def update_stats(self):
        """Update statistics display"""
        self.stat_fps.config(text=str(self.fps))
        self.stat_frames.config(text=str(self.frame_count))
        self.stat_violations.config(text=str(self.total_violations))
        self.stat_safe.config(text=str(self.total_safe))
        
    def set_live_status(self, active, text="READY"):
        """Set live status indicator"""
        if active:
            self.live_dot.config(fg=self.colors['accent_green'])
            self.live_label.config(text=f" {text}", fg=self.colors['accent_green'])
        else:
            self.live_dot.config(fg=self.colors['text_secondary'])
            self.live_label.config(text=f" {text}", fg=self.colors['text_secondary'])
        
    def detect_image(self):
        """Detect helmets in an image"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        
        if not file_path:
            return
            
        self.status_label.config(text="Processing image...")
        self.set_live_status(True, "PROCESSING")
        self.root.update()
        
        try:
            # Run detection
            results = self.model.predict(
                source=file_path,
                conf=self.confidence_var.get(),
                imgsz=640,
                save=False,
                verbose=False
            )
            
            # Get annotated image
            annotated = results[0].plot()
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            
            # Display
            self.display_image(annotated_rgb)
            
            # Count detections
            violations = 0
            safe = 0
            for box in results[0].boxes:
                cls_name = self.model.names[int(box.cls[0])]
                if 'without' in cls_name.lower() or 'no' in cls_name.lower():
                    violations += 1
                else:
                    safe += 1
            
            self.total_violations += violations
            self.total_safe += safe
            self.frame_count += 1
            self.update_stats()
            
            self.set_live_status(False, "COMPLETE")
            self.status_label.config(text=f"Detected: {violations} violations, {safe} safe")
            
        except Exception as e:
            messagebox.showerror("Error", f"Detection failed: {str(e)}")
            self.set_live_status(False, "ERROR")
            
    def detect_video(self):
        """Detect helmets in a video file"""
        file_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv")]
        )
        
        if not file_path:
            return
            
        self.stop_all()
        self.reset_stats()
        
        self.is_video_running = True
        self.btn_video.set_state(False)
        self.btn_image.set_state(False)
        self.btn_webcam_start.set_state(False)
        self.btn_webcam_stop.set_state(True)
        
        self.set_live_status(True, "VIDEO")
        
        # Start video thread
        self.video_thread = threading.Thread(
            target=self.process_video_optimized,
            args=(file_path,),
            daemon=True
        )
        self.video_thread.start()
        
    def process_video_optimized(self, video_path):
        """Process video file - HEAVILY OPTIMIZED for 40+ FPS"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            self.is_video_running = False
            return
            
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        self.status_label.config(text=f"Processing video... 0/{total_frames}")
        
        frame_counter = 0
        process_counter = 0
        fps_counter = 0
        fps_start_time = time.time()
        last_detections = []
        
        while self.is_video_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_counter += 1
            fps_counter += 1
            
            # Calculate FPS every second
            current_time = time.time()
            if current_time - fps_start_time >= 1.0:
                self.fps = fps_counter
                fps_counter = 0
                fps_start_time = current_time
            
            # Resize frame for faster processing
            h, w = frame.shape[:2]
            scale = min(self.process_width / w, self.process_height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            # Process detection only every N frames
            if frame_counter % self.process_skip == 0:
                process_counter += 1
                self.frame_count = process_counter
                
                # Resize for detection
                small_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                # Run detection with optimized settings
                results = self.model(
                    small_frame,
                    conf=self.confidence_var.get(),
                    imgsz=self.process_width,
                    verbose=False,
                    half=False  # CPU doesn't support half precision
                )[0]
                
                # Store detections (scaled back to original size)
                last_detections = []
                for box in results.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Scale coordinates back
                    x1 = int(x1 / scale)
                    y1 = int(y1 / scale)
                    x2 = int(x2 / scale)
                    y2 = int(y2 / scale)
                    
                    cls_name = self.model.names[cls]
                    is_violation = 'without' in cls_name.lower() or 'no' in cls_name.lower()
                    
                    if is_violation:
                        self.total_violations += 1
                    else:
                        self.total_safe += 1
                        
                    last_detections.append((x1, y1, x2, y2, conf, is_violation))
            
            # Draw detections on current frame (even if not processed this frame)
            display_frame = frame.copy()
            for (x1, y1, x2, y2, conf, is_violation) in last_detections:
                if is_violation:
                    color = (0, 0, 255)  # Red
                    text = f'NO HELMET {conf:.0%}'
                else:
                    color = (0, 255, 0)  # Green
                    text = f'HELMET {conf:.0%}'
                
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                
                # Background for text
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(display_frame, (x1, y1-th-10), (x1+tw+10, y1), color, -1)
                cv2.putText(display_frame, text, (x1+5, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add overlay info
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (10, 10), (200, 80), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
            
            cv2.putText(display_frame, f'FPS: {self.fps}', (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_frame, f'Frame: {frame_counter}/{total_frames}', (20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Queue frame for display
            if not self.frame_queue.full():
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                try:
                    self.frame_queue.put_nowait(frame_rgb)
                except queue.Full:
                    pass
            
            # Update stats periodically
            if frame_counter % 30 == 0:
                self.root.after(0, self.update_stats)
                progress = int((frame_counter / total_frames) * 100)
                self.root.after(0, lambda p=progress, f=frame_counter, t=total_frames: 
                              self.status_label.config(text=f"Processing: {p}% ({f}/{t})"))
        
        cap.release()
        self.is_video_running = False
        
        self.root.after(0, self.on_video_complete)
        
    def on_video_complete(self):
        """Called when video processing completes"""
        self.btn_video.set_state(True)
        self.btn_image.set_state(True)
        self.btn_webcam_start.set_state(True)
        self.btn_webcam_stop.set_state(False)
        
        self.set_live_status(False, "COMPLETE")
        self.status_label.config(text=f"Complete! {self.total_violations} violations, {self.total_safe} safe")
        self.update_stats()
        
    def start_webcam(self):
        """Start webcam detection"""
        self.stop_all()
        self.reset_stats()
        
        self.is_webcam_running = True
        self.btn_webcam_start.set_state(False)
        self.btn_webcam_stop.set_state(True)
        self.btn_video.set_state(False)
        self.btn_image.set_state(False)
        
        self.set_live_status(True, "LIVE")
        
        # Start webcam thread
        self.video_thread = threading.Thread(
            target=self.run_webcam_optimized,
            daemon=True
        )
        self.video_thread.start()
        
    def run_webcam_optimized(self):
        """Run webcam detection - OPTIMIZED for 40+ FPS"""
        self.cap = cv2.VideoCapture(0)
        
        # Set camera properties for speed
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for real-time
        
        if not self.cap.isOpened():
            self.root.after(0, lambda: messagebox.showerror("Error", "Cannot open webcam"))
            self.stop_all()
            return
            
        self.status_label.config(text="Webcam running - optimized mode")
        
        frame_counter = 0
        fps_counter = 0
        fps_start_time = time.time()
        last_detections = []
        
        while self.is_webcam_running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_counter += 1
            fps_counter += 1
            
            # Calculate FPS
            current_time = time.time()
            if current_time - fps_start_time >= 1.0:
                self.fps = fps_counter
                fps_counter = 0
                fps_start_time = current_time
            
            # Get frame dimensions
            h, w = frame.shape[:2]
            scale = min(self.process_width / w, self.process_height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            # Process detection only every N frames
            if frame_counter % self.process_skip == 0:
                self.frame_count += 1
                
                # Resize for detection
                small_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                # Run detection
                results = self.model(
                    small_frame,
                    conf=self.confidence_var.get(),
                    imgsz=self.process_width,
                    verbose=False
                )[0]
                
                # Store detections
                last_detections = []
                for box in results.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Scale coordinates back
                    x1 = int(x1 / scale)
                    y1 = int(y1 / scale)
                    x2 = int(x2 / scale)
                    y2 = int(y2 / scale)
                    
                    cls_name = self.model.names[cls]
                    is_violation = 'without' in cls_name.lower() or 'no' in cls_name.lower()
                    
                    if is_violation:
                        self.total_violations += 1
                    else:
                        self.total_safe += 1
                        
                    last_detections.append((x1, y1, x2, y2, conf, is_violation))
            
            # Draw detections
            display_frame = frame.copy()
            for (x1, y1, x2, y2, conf, is_violation) in last_detections:
                if is_violation:
                    color = (0, 0, 255)
                    text = f'NO HELMET'
                else:
                    color = (0, 255, 0)
                    text = f'HELMET'
                
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)
                
                # Text background
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(display_frame, (x1, y1-th-10), (x1+tw+10, y1), color, -1)
                cv2.putText(display_frame, text, (x1+5, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Overlay stats
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (10, 10), (220, 110), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
            
            cv2.putText(display_frame, f'FPS: {self.fps}', (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f'Violations: {self.total_violations}', (20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
            cv2.putText(display_frame, f'Safe: {self.total_safe}', (20, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Queue frame
            if not self.frame_queue.full():
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                try:
                    self.frame_queue.put_nowait(frame_rgb)
                except queue.Full:
                    pass
            
            # Update stats
            if frame_counter % 15 == 0:
                self.root.after(0, self.update_stats)
        
        if self.cap:
            self.cap.release()
            
    def stop_webcam(self):
        """Stop webcam/video"""
        self.stop_all()
        
    def stop_all(self):
        self.is_webcam_running = False
        self.is_video_running = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
            
        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        self.btn_webcam_start.set_state(True)
        self.btn_webcam_stop.set_state(False)
        self.btn_video.set_state(True)
        self.btn_image.set_state(True)
        
        self.set_live_status(False, "STOPPED")
        self.status_label.config(text="Stopped")
        
    def update_display_loop(self):
        """Continuous display update loop"""
        try:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get_nowait()
                self.display_image(frame)
        except queue.Empty:
            pass
        

        self.root.after(16, self.update_display_loop)
        
    def display_image(self, img_array):
        """Display image on canvas"""
        try:
            # Get canvas size
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width <= 1:
                canvas_width = 800
            if canvas_height <= 1:
                canvas_height = 600
            
            # Calculate aspect-ratio preserving resize
            h, w = img_array.shape[:2]
            aspect = w / h
            
            if canvas_width / canvas_height > aspect:
                new_height = canvas_height
                new_width = int(canvas_height * aspect)
            else:
                new_width = canvas_width
                new_height = int(canvas_width / aspect)
            
            # Resize using fast interpolation
            img_resized = cv2.resize(img_array, (new_width, new_height), 
                                    interpolation=cv2.INTER_LINEAR)
            
            # Convert to PhotoImage
            img_pil = Image.fromarray(img_resized)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            
            # Update canvas
            self.canvas.delete("all")
            self.canvas.create_image(
                canvas_width // 2,
                canvas_height // 2,
                image=img_tk,
                anchor=tk.CENTER
            )
            self.canvas.image = img_tk
            
        except Exception:
            pass
        
    def on_closing(self):
        """Handle window closing"""
        self.stop_all()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    
    # Set DPI awareness for sharper rendering on Windows
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass
    
    app = HelmetDetectionGUI(root)
    root.mainloop()
