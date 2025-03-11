import cv2
import time
import signal
import os
from threading import Thread
from multiprocessing import Process, Queue, Event, set_start_method

class UsbCam:
    def __init__(self, camera_dict, width=1280, height=480, fps=30, visualize=True):
        self.camera_names = list(camera_dict.keys())  # preserve order
        self.camera_dict = camera_dict
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
        for name, cam_id in camera_dict.items():
            process = Process(
                target=self._camera_worker,
                args=(name, cam_id, self.frame_queues[name], self.stop_event, self.width, self.height, self.fps),
            )
            self.processes.append(process)

        

    @staticmethod
    def _camera_worker(name, cam_id, frame_queue, stop_event, width, height, fps):
        """Worker process for a single camera."""
        print(f"[INFO] Initializing camera worker for {name}...")
        dt = 1 / fps
        
        while not stop_event.is_set():
            # Attempt to connect to the camera            
            cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
            
            if not cap.isOpened():
                print(f"[ERROR] Camera {name} failed to open.")
                cap.release()
                time.sleep(1)
                continue
            print(f"[INFO] Camera {name} started.")
            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

            # bright_setting_done = cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.0)   # [-64~64] 범위
            # contrast_setting_done = cap.set(cv2.CAP_PROP_CONTRAST, 32.0)     # [0~64] 범위
            # saturation_setting_done = cap.set(cv2.CAP_PROP_SATURATION, 78.99999)   # [0, 79) 범위
            # sharpness_setting_done = cap.set(cv2.CAP_PROP_SHARPNESS, 7.99999) # [0, 7.99)
            
            # print(f'{self.camera_name} bright_setting_done : {bright_setting_done}')
            # print(f'{self.camera_name} contrast_setting_done : {contrast_setting_done}')
            # print(f'{self.camera_name} saturation_setting_done : {saturation_setting_done}')
            # print(f'{self.camera_name} sharpness_setting_done : {sharpness_setting_done}')
    
            while not stop_event.is_set():
                t_start = time.time()
                ret, frame = cap.read()
                
                while not stop_event.is_set() and frame is None:
                    # If the camera fails, retry connection
                    if not ret:
                        print(f"[ERROR] Camera {name} failed to capture frame.")
                        break
                    
                    ret, frame = cap.read()
                    print(f'[DEBUG] frame: {frame}')
                
                # Send frame to queue (non-blocking)
                
                try:
                    # Remove the oldest frame if the queue is full
                    if frame_queue.full():
                        discarded_frame = frame_queue.get()  # Remove the oldest frame
                        # print(f"[INFO] Discarded oldest frame for {name}")

                    # Add the new frame to the queue
                    frame_queue.put_nowait(frame)
                except Exception as e:
                    print(f"[ERROR] Failed to add frame to queue for {name}: {e}")
                # try:
                #     frame_queue.put_nowait(frame)
                # except:
                #     print(f"[WARNING] Frame queue for {name} is full. Dropping frame.")

                
                
                t_end = time.time()
                t_sleep = max(dt - (t_end - t_start), 0)
                # print(f'[DEBUG] camera {cam_id} t_sleep: {t_sleep}')
                if t_sleep == 0:
                    print(f'[WARNING] camera loop is delaying. Elapsed time for a loop: {t_end - t_start}')
                    
                time.sleep(t_sleep)

            cap.release()
            print(f"[INFO] Camera {name} connection lost. Retrying...")
    
    def visualize(self):
        """Display camera frames in real-time."""
        print(f"Press 'q' to quit visualization.")
        time_show = time.time()
        try:
            while not self.stop_event.is_set():
                
                frames = self.get_frames(self.camera_names)
                for name, frame in frames.items():
                    if frame is not None:
                        if name == 'head_camera':
                            print(f"head cam dt: {time.time() - time_show}")
                            time_show = time.time()
                            
                        cv2.imshow(name, frame)
                        
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop_event.set()
                    print("[INFO] 'q' pressed. Exiting camera frame visualization.")
                    break
        except KeyboardInterrupt:
            self.stop_event.set()
            print("[INFO] Ctrl+C detected. Exiting visualization.")
        finally:
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
        self.stop_event.set()  # Signal all processes to stop
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
            if not queue.empty():
                if name in camera_names:
                    frames[name] = queue.get()
            else:
                if name in camera_names:
                    frames[name] = None
        return frames
    
    
def signal_handler(sig, frame):
    """Handle Ctrl+C and propagate to child processes."""
    print("[INFO] Ctrl+C detected. Exiting...")
    usb_cam.stop()
    os._exit(0)  # Forcefully terminate all processes

if __name__ == "__main__":
    try:
        # Use spawn start method for multiprocessing
        set_start_method("spawn", force=True)
        
        # Attach the signal handler for Ctrl+C
        signal.signal(signal.SIGINT, signal_handler)
        
        # Example usage:
        camera_dict = {
            'lhand_camera': 1,
            'rhand_camera': 2,
            'head_camera': 0,
        }
        usb_cam = UsbCam(camera_dict, width=1280, height=480, fps=30, visualize=True)
        # usb_cam.read_and_show()
        usb_cam.start()
        
        while True:
            user_input = input()
            if user_input.lower() == 'q':
                usb_cam.stop()
                break

    except KeyboardInterrupt:
        print("[INFO] Program interrupted by user (Ctrl+C). Exiting...")
        # usb_cam.stop()
    except Exception as e:
        print(f"[ERROR] Exception occurred: {e}")
    finally:
        try:
            usb_cam.stop()
        except NameError:
            pass

    
