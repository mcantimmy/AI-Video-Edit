# ui_features.py
import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import os
import threading
import time
import numpy as np
from PIL import Image, ImageTk
from Video_Content_Analysis import VideoContentAnalyzer, VideoEditorAI

class VideoEditorUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Video Editor")
        self.root.geometry("1200x800")
        
        # Initialize analyzer and editor
        self.analyzer = VideoContentAnalyzer()
        self.editor = VideoEditorAI(self.analyzer)
        
        # Video path and edit decisions
        self.video_path = None
        self.edit_decisions = []
        self.preview_playing = False
        self.current_frame = 0
        
        self.setup_ui()
        
    def setup_ui(self):
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # File selection
        file_frame = ttk.LabelFrame(main_frame, text="Video File")
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(file_frame, text="Select Video", command=self.select_video).pack(side=tk.LEFT, padx=5, pady=5)
        self.file_label = ttk.Label(file_frame, text="No file selected")
        self.file_label.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        
        # Preview area
        preview_frame = ttk.LabelFrame(main_frame, text="Preview")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.preview_canvas = tk.Canvas(preview_frame, bg="black")
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Controls
        controls_frame = ttk.Frame(preview_frame)
        controls_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(controls_frame, text="▶️ Play/Pause", command=self.toggle_preview).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Generate Timeline", command=self.generate_timeline).pack(side=tk.LEFT, padx=5)
        
        # Timeline visualization
        timeline_frame = ttk.LabelFrame(main_frame, text="Timeline")
        timeline_frame.pack(fill=tk.X, pady=5)
        
        self.timeline_canvas = tk.Canvas(timeline_frame, height=100, bg="white")
        self.timeline_canvas.pack(fill=tk.X, padx=5, pady=5)
        
        # Scrollbar for timeline
        scrollbar = ttk.Scrollbar(timeline_frame, orient="horizontal", command=self.timeline_canvas.xview)
        scrollbar.pack(fill=tk.X)
        self.timeline_canvas.configure(xscrollcommand=scrollbar.set)
        
    def select_video(self):
        """Open file dialog to select a video file"""
        self.video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=(("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*"))
        )
        
        if self.video_path:
            self.file_label.config(text=os.path.basename(self.video_path))
            self.load_video_preview()
        
    def load_video_preview(self):
        """Load the first frame of the video for preview"""
        if not self.video_path:
            return
        
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Convert OpenCV BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize to fit canvas
            canvas_width = self.preview_canvas.winfo_width()
            canvas_height = self.preview_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                frame_rgb = cv2.resize(frame_rgb, (canvas_width, canvas_height))
            
            # Convert to PhotoImage
            self.preview_image = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
            self.preview_canvas.create_image(0, 0, image=self.preview_image, anchor=tk.NW)
    
    def generate_timeline(self):
        """Generate and visualize timeline from edit decisions"""
        if not self.video_path:
            return
        
        # Clear existing timeline
        self.timeline_canvas.delete("all")
        
        # Generate edit decisions if not already done
        if not self.edit_decisions:
            self.edit_decisions = self.editor.create_rough_cut(self.video_path)
        
        # Get video duration
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = total_frames / fps
        cap.release()
        
        # Set timeline width
        timeline_width = max(1000, self.timeline_canvas.winfo_width())
        self.timeline_canvas.config(scrollregion=(0, 0, timeline_width, 100))
        
        # Draw timeline base
        self.timeline_canvas.create_rectangle(
            10, 40, timeline_width-10, 60, 
            fill="lightgray", outline="gray"
        )
        
        # Draw timeline markers
        for i in range(0, int(total_duration) + 1, 10):
            x_pos = 10 + (i / total_duration) * (timeline_width - 20)
            self.timeline_canvas.create_line(x_pos, 30, x_pos, 70, fill="gray")
            self.timeline_canvas.create_text(x_pos, 20, text=f"{i}s")
        
        # Draw edit decisions
        for i, decision in enumerate(self.edit_decisions):
            start_x = 10 + (decision["start_time"] / total_duration) * (timeline_width - 20)
            end_x = 10 + (decision["end_time"] / total_duration) * (timeline_width - 20)
            
            # Color based on emotion
            color = self.get_emotion_color(decision["emotion"])
            
            # Draw clip
            self.timeline_canvas.create_rectangle(
                start_x, 40, end_x, 60, 
                fill=color, outline="black",
                tags=f"clip_{i}"
            )
            
            # Add tooltip
            self.timeline_canvas.tag_bind(
                f"clip_{i}", "<Enter>", 
                lambda event, d=decision: self.show_clip_tooltip(event, d)
            )
            self.timeline_canvas.tag_bind(
                f"clip_{i}", "<Leave>", 
                self.hide_clip_tooltip
            )
    
    def get_emotion_color(self, emotion):
        """Map emotion to color"""
        colors = {
            "joy": "#FFDD00",
            "sadness": "#6699CC",
            "anger": "#CC3333",
            "fear": "#9966CC",
            "surprise": "#FF9933",
            "neutral": "#CCCCCC"
        }
        return colors.get(emotion, "#CCCCCC")
    
    def show_clip_tooltip(self, event, decision):
        """Show tooltip with clip information"""
        x, y = event.x, event.y
        
        # Create tooltip
        self.tooltip = tk.Toplevel(self.root)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.geometry(f"+{x+self.root.winfo_x()+10}+{y+self.root.winfo_y()+10}")
        
        # Add content
        frame = ttk.Frame(self.tooltip, relief="solid", borderwidth=1)
        frame.pack(fill="both", expand=True)
        
        ttk.Label(frame, text=f"Time: {decision['start_time']:.1f}s - {decision['end_time']:.1f}s").pack(anchor="w")
        ttk.Label(frame, text=f"Emotion: {decision['emotion']}").pack(anchor="w")
        ttk.Label(frame, text=f"Content: {decision['content'][:50]}...").pack(anchor="w")
    
    def hide_clip_tooltip(self, event):
        """Hide the tooltip"""
        if hasattr(self, 'tooltip'):
            self.tooltip.destroy()
    
    def toggle_preview(self):
        """Play/pause the preview"""
        if not self.video_path:
            return
        
        self.preview_playing = not self.preview_playing
        
        if self.preview_playing:
            # Start playback in a separate thread
            self.playback_thread = threading.Thread(target=self.play_preview)
            self.playback_thread.daemon = True
            self.playback_thread.start()
        
    def play_preview(self):
        """Play video preview"""
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = 1 / fps
        
        if self.current_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        
        while self.preview_playing:
            ret, frame = cap.read()
            if not ret:
                self.preview_playing = False
                break
            
            # Convert frame and display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize to fit canvas
            canvas_width = self.preview_canvas.winfo_width()
            canvas_height = self.preview_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                frame_rgb = cv2.resize(frame_rgb, (canvas_width, canvas_height))
            
            # Update preview
            preview_image = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
            
            # Update on main thread
            self.root.after(0, lambda img=preview_image: self.update_preview(img))
            
            self.current_frame += 1
            time.sleep(delay)
        
        cap.release()
    
    def update_preview(self, image):
        """Update preview canvas with new frame"""
        self.preview_image = image  # Keep reference
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(0, 0, image=self.preview_image, anchor=tk.NW)

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = VideoEditorUI(root)
    root.mainloop()