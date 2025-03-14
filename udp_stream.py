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
        self.camera_names = camera_dict  # preserve order
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
        self.sock.settimeout(1.0)
        
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        if mode == "recv":
            print(f"[{id}] UDP receiver node initialized")
            try:
                self.sock.bind((host, port))
            except OSError as e:
                print(f"[{id}] Socket binding failed: {e}")
                self.sock.close()
                raise
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
            self.recv_thread.join(timeout=2)
            if self.recv_thread.is_alive():
                print(f"[{self.id}] Force closing socket")
            self.sock.close()
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

    def send_frame_with_rtt(self, frame):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, buffer = cv2.imencode(".jpg", frame, encode_param)

        seq_num = self.sequence_number
        self.sequence_number += 1

        seq_bytes = struct.pack('I', seq_num)

        self.sock.sendto(seq_bytes + buffer.tobytes(), (self.host, self.port))

        send_time = time.time()

        try:
            self.sock.settimeout(1.0)
            ack_data, _ = self.sock.recvfrom(1024)

            ack_seq = struct.unpack('I', ack_data)[0]

            if ack_seq == seq_num:
                rtt = (time.time() - send_time) * 1000 
                print(f"[{self.id}] RTT: {rtt:.2f} ms (seq {seq_num})")
            else:
                print(f"[{self.id}] Sequence mismatch: expected {seq_num}, got {ack_seq}")

        except socket.timeout:
            print(f"[{self.id}] Timeout waiting for ACK for seq {seq_num}")

    def recv_frame(self):
        while self.running:
            try:
                data, _ = self.sock.recvfrom(65536)
                np_arr = np.frombuffer(data, dtype=np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                self.frame = cv2.resize(frame, (1280, 480))
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[{self.id}] Error receiving frame: {e}")
                break
            
    def recv_frame_with_ack(self):
        while self.running:
            try:
                data, sender_addr = self.sock.recvfrom(65536)

                seq_bytes = data[:4]
                seq_num = struct.unpack('I', seq_bytes)[0]
                frame_data = data[4:]

                self.sock.sendto(seq_bytes, sender_addr)

                np_arr = np.frombuffer(frame_data, dtype=np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                self.frame = cv2.resize(frame, (1280, 480))

            except socket.timeout:
                continue
            except Exception as e:
                print(f"[{self.id}] Error receiving frame: {e}")
                continue
            
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
        frame = cv2.resize(frame, (1280, 480))
        return frame

    def close_camera(self):
        self.cap.release()
        print(f"[{self.id}] Camera closed")

    def __del__(self):
        """Destructor to clean up socket"""
        if hasattr(self, 'sock'):
            self.running = False
            try:
                self.sock.close()
            except:
                pass
            