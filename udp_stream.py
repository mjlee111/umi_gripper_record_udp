import cv2
import socket
import struct
import numpy as np
import threading
from multiprocessing import Process, Queue, Event, set_start_method
import time
from threading import Thread

class UDPCamera:
    def __init__(self, camera_dict, ip_dict, width=1280, height=480, fps=30, visualize=True):
        self.camera_names = list(camera_dict.keys())  # preserve order
        self.camera_dict = camera_dict
        self.ip_dict = ip_dict
        self.width = width
        self.height = height
        self.fps = fps
        self.dt = 1/fps
        self.visualize_flag = visualize
        self.visualize_thread = None
        
        # Create queues and processes for each camera
        self.frame_queues = {name: Queue(maxsize=2) for name in self.camera_names}
        self.stop_event = Event()
        self.processes = []
        
        # Initialize processes for each camera
        for name in self.camera_names:
            process = Process(
                target=self._camera_worker,
                args=(name, self.frame_queues[name], self.stop_event, 
                      self.width, self.height, self.fps, ip_dict[name]),
            )
            self.processes.append(process)

    @staticmethod
    def _camera_worker(name, frame_queue, stop_event, width, height, fps, ip_info):
        """Worker process for a single camera with UDP receiving."""
        print(f"[INFO] Initializing UDP receiver for {name}...")
        dt = 1 / fps
        
        # Initialize UDP stream
        host, port = ip_info
        udp_stream = UDPStream(name, host, port, mode="recv")
        udp_stream.start_recv_thread()
        
        while not stop_event.is_set():
            t_start = time.time()
            
            # Get frame from UDP stream
            frame = udp_stream.frame
            
            if frame is not None:
                # Add frame to queue for visualization
                try:
                    if frame_queue.full():
                        _ = frame_queue.get()
                    frame_queue.put_nowait(frame)
                except Exception as e:
                    print(f"[ERROR] Failed to add frame to queue for {name}: {e}")
            
            t_end = time.time()
            t_sleep = max(dt - (t_end - t_start), 0)
            if t_sleep == 0:
                print(f'[WARNING] Camera {name} loop is delaying. Elapsed time: {t_end - t_start}')
            
            time.sleep(t_sleep)
        
        udp_stream.stop_recv_thread()
        print(f"[INFO] UDP receiver {name} stopped.")

    def visualize(self):
        """Display camera frames in real-time."""
        print("Press 'q' to quit visualization.")
        while not self.stop_event.is_set():
            frames = self.get_frames(self.camera_names)
            for name, frame in frames.items():
                if frame is not None:
                    cv2.imshow(name, frame)
                    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_event.set()
                break
                
        cv2.destroyAllWindows()

    def start(self):
        """Start all camera processes."""
        for process in self.processes:
            process.start()

        if self.visualize_flag:
            self.visualize_thread = Thread(target=self.visualize, daemon=True)
            self.visualize_thread.start()

    def stop(self):
        """Stop all camera processes."""
        print("[INFO] Stopping all cameras...")
        self.stop_event.set()
        for process in self.processes:
            process.terminate()
            process.join()

        if self.visualize_flag and self.visualize_thread:
            self.visualize_thread.join()
            cv2.destroyAllWindows()
        
        print("[INFO] All camera processes stopped.")

    def get_frames(self, camera_names):
        """Retrieve frames from all cameras."""
        frames = {}
        for name, queue in self.frame_queues.items():
            if not queue.empty() and name in camera_names:
                frames[name] = queue.get()
            elif name in camera_names:
                frames[name] = None
        return frames

class UDPStream:
    def __init__(self, id, host, port, mode="send"):
        self.id = id
        self.host = host
        self.port = port
        self.mode = mode
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        if mode == "recv":
            print(f"[{id}] UDP receiver node initialized")
            self.sock.bind((host, port))
            self.frame = None
            self.recv_thread = None
            self.running = False
        else:
            print(f"[{id}] UDP sender node initialized")

    def start_recv_thread(self):
        print(f"[{self.id}] Starting UDP receiver thread")
        self.running = True
        self.recv_thread = threading.Thread(target=self.recv_frame)
        self.recv_thread.start()
        
    def stop_recv_thread(self):
        print(f"[{self.id}] Stopping UDP receiver thread")
        if self.recv_thread:
            self.running = False
            self.recv_thread.join()
            self.recv_thread = None
            
    def start_send_thread(self):
        print(f"[{self.id}] Starting UDP sender thread")
        self.running = True
        self.send_thread = threading.Thread(target=self.stream_frame)
        self.send_thread.start()

    def stop_send_thread(self):
        print(f"[{self.id}] Stopping UDP sender thread")
        if self.send_thread:
            self.running = False
            self.send_thread.join()
            self.send_thread = None
            
    def stream_frame(self):
        while self.running:
            frame = self.get_frame()
            if frame is not None:
                frame = cv2.resize(frame, (640, 240))
                self.send_frame(frame)

    def send_frame(self, frame):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, buffer = cv2.imencode(".jpg", frame, encode_param)
        self.sock.sendto(buffer.tobytes(), (self.host, self.port))

    def recv_frame(self):
        while self.running:
            data, _ = self.sock.recvfrom(65536)
            np_arr = np.frombuffer(data, dtype=np.uint8)
            self.frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def open_camera(self, camera_id=0, width=1280, height=480, fps=15):
        print(f"[{self.id}] Trying to open camera {camera_id}")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print(f"[{self.id}] Failed to open camera.")
            exit()

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        print(f"[{self.id}] Camera {camera_id} opened with {width}x{height} at {fps} fps")
        
    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print(f"[{self.id}] Failed to read frame.")
            return None
        return frame

    def close_camera(self):
        self.cap.release()
        print(f"[{self.id}] Camera closed")
            
if __name__ == "__main__":
    set_start_method("spawn", force=True)
    
    camera_dict = {
        'lhand_camera': None, 
        'rhand_camera': None,
    }
    
    ip_dict = {
        'lhand_camera': ('0.0.0.0', 5000), 
        'rhand_camera': ('0.0.0.0', 5001),
    }
    
    udp_camera_names = list(camera_dict.keys())
    
    udp_cam = UDPCamera(camera_dict, ip_dict, width=1280, height=480, fps=30, visualize=True)
    udp_cam.start()
    
    try:
        while True:
            if input().lower() == 'q':
                break
    finally:
        udp_cam.stop()