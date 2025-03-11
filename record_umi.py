from vive_ultimate_tracker import ViveUTracker
from usb_cam import UsbCam
import threading
import xr
import time
import numpy as np
from constants import *
from tqdm import tqdm
import h5py
import cv2
import argparse
from multiprocessing import set_start_method
import signal
import os
from udp_stream import UDPCamera


def main(args):
    task_config = TASK_CONFIGS[args['task_name']]
    dataset_dir = task_config['dataset_dir']
    max_timesteps = task_config['episode_len']
    usb_camera_names = task_config['usb_camera_names']
    udp_camera_names = task_config['udp_camera_dict'].keys()
    tracker_names = task_config['tracker_names']
    
    global stop_record_flag, stop_user_input_thread_flag, viveUTracker, usb_cam
    stop_record_flag = False
    stop_user_input_thread_flag = False
    
    # Use spawn start method for multiprocessing
    set_start_method("spawn", force=True)
    
    # Attach the signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    ### CAMERA PROCESS ###
    camera_dict = {}
    
    for id, cam_name in enumerate(usb_camera_names):
        camera_dict[cam_name] = id
    
    print(f'USB Camera_dict: {camera_dict}')

    usb_cam = UsbCam(camera_dict, width=1280, height=480, fps=30, visualize=False)
    usb_cam.start()
    print(f'USB Camera Processes Start!')

    ### UDP CAMERA PROCESS ###
    udp_camera_dict = task_config['udp_camera_dict']
    print(f'UDP Camera_dict: {udp_camera_dict}')
    ip_dict = task_config['ip_dict']
    print(f'UDP IP_dict: {ip_dict}')

    udp_cam = UDPCamera(udp_camera_dict, ip_dict, width=1280, height=480, fps=30, visualize=False)
    udp_cam.start()
    print(f'UDP Camera Processes Start!')    
    
    ### VIVE TRACKER THREAD ###
    viveUTracker = ViveUTracker(
        instance_create_info=xr.InstanceCreateInfo(
            enabled_extension_names=[
                # A graphics extension is mandatory (without a headless extension)
                xr.MND_HEADLESS_EXTENSION_NAME,
                xr.extension.HTCX_vive_tracker_interaction.NAME,
            ],
        ),
        HZ = 2*HZ,
        visualize = False,
    )
    viveUTracker.thread_start()
    
    if args['episode_idx'] is not None:
        episode_idx = args['episode_idx']
    else:
        episode_idx = get_auto_index(dataset_dir)
    overwrite = True
    
    dataset_name = f'episode_{episode_idx}'
    print(args['task_name'] + ', ' + dataset_name + '\n' )
    
    # saving dataset
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    dataset_path = os.path.join(dataset_dir, dataset_name)
    if os.path.isfile(dataset_path) and not overwrite:
        print(f'Dataset already exist at \n{dataset_path}\nHint: set overwrite to True.')
        exit()
    
    data_dict = {}
    for tracker_name in tracker_names:
        data_dict[f'/observations/tracker_poses/{tracker_name}'] = []
    
    for cam_name in usb_camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []
        
    for cam_name in udp_camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []

    images = dict() 
    
    # check image streaming
    images = usb_cam.get_frames(usb_camera_names)
    while any(images[camera_name] is None for camera_name in usb_camera_names):
        print('waiting usb camera streaming...')
        time.sleep(1.0)
        images = usb_cam.get_frames(usb_camera_names)
    
    # check image streaming
    images = udp_cam.get_frames(udp_camera_names)
    while any(images[camera_name] is None for camera_name in udp_camera_names):
        print('waiting udp camera streaming...')
        time.sleep(1.0)
        images = udp_cam.get_frames(udp_camera_names)

    # check tracker streaming
    tracker_poses = viveUTracker.get_tracker_poses(tracker_names)
    while tracker_poses is None:
        print('waiting tracker streaming...')
        time.sleep(1.0)
        tracker_poses = viveUTracker.get_tracker_poses(tracker_names)
    
    print('Are you ready for DATA COLLECTION?')
    for cnt in range(3):
        print(3 - cnt, 'seconds before to start DATA COLLECTION!!!')
        time.sleep(1.0)
        
    print('DATA COLLECTION Start!')
    data_tick = 0
    time_begin = time.perf_counter()
    next_loop_time = time.perf_counter()
    
    for t in tqdm(range(max_timesteps)):
        time_start = time.perf_counter()    
        
        if stop_record_flag:
            print('Data Recording is Stopped!')
            break
        
        # Get tracker poses and frames
        tracker_poses = viveUTracker.get_tracker_poses(tracker_names)
        images = usb_cam.get_frames(usb_camera_names)
        images.update(udp_cam.get_frames(udp_camera_names))
        
        # Append tracker poses
        for tracker_name in tracker_names:
            if tracker_name in tracker_poses:
                data_dict[f'/observations/tracker_poses/{tracker_name}'].append(tracker_poses[tracker_name])
            else:
                print(f"[WARNING] Missing tracker data for '{tracker_name}'. Skipping...")
                
        # Append camera frames
        for cam_name, image in images.items():
            while not stop_record_flag and image is None:
                time.sleep(0.001)
                image = usb_cam.get_frames([cam_name])[cam_name]
                
            data_dict[f'/observations/images/{cam_name}'].append(image)
    
        data_tick += 1
        
        loop_duration = 1 / HZ
        next_loop_time += loop_duration
        time_elapsed = time.perf_counter() - time_start
        time_sleep = max(next_loop_time - time.perf_counter(), 0)   
        
        if time_sleep == 0:
            print(f'[WARNING] Data collection is too slow [{time_elapsed} s]')
            
        time.sleep(max(next_loop_time - time.perf_counter(), 0))
        
    print(f'Total Data Record Time [s]: {time.perf_counter() - time_begin}')
    print(f'Avg Record Frequency [Hz]: {data_tick / (time.perf_counter() - time_begin)}')    
    
    viveUTracker.thread_stop()
    usb_cam.stop()
    udp_cam.stop()
    # stop_user_input_thread_flag = True    
    
    COMPRESS = True
    
    if COMPRESS:
        # JPEG compression
        t0 = time.time()
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50] # tried as low as 20, seems fine
        compressed_image_len = []
        for cam_name in usb_camera_names+udp_camera_names:
            image_list = data_dict[f'/observations/images/{cam_name}']
            compressed_list = []
            compressed_image_len.append([])
            
            for image in image_list:
                result, encoded_image = cv2.imencode('.jpg', image, encode_param)
                compressed_list.append(encoded_image)
                compressed_image_len[-1].append(len(encoded_image)) 
                
            data_dict[f'/observations/images/{cam_name}'] = compressed_list
            
        print(f'compression: {time.time() - t0:.2f}s')

        # pad so it has same length
        t0 = time.time()
        compressed_image_len = np.array(compressed_image_len)
        padded_size = compressed_image_len.max()
        for cam_name in usb_camera_names+udp_camera_names:
            compressed_image_list = data_dict[f'/observations/images/{cam_name}']
            padded_compressed_image_list = []
            for compressed_image in compressed_image_list:
                padded_compressed_image = np.zeros(padded_size, dtype='uint8')
                image_len = len(compressed_image)
                padded_compressed_image[:image_len] = compressed_image
                padded_compressed_image_list.append(padded_compressed_image)
            data_dict[f'/observations/images/{cam_name}'] = padded_compressed_image_list
        print(f'padding: {time.time() - t0:.2f}s')
        
    # HDF5
    t0 = time.time()
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
        # root.attrs['sim'] = False
        root.attrs['compress'] = COMPRESS
        root.attrs['Hz'] = HZ
        obs = root.create_group('observations')
        image = obs.create_group('images')
        tracker_pose = obs.create_group('tracker_poses')
        # depth = obs.create_group('depth_images')
        for cam_name in usb_camera_names+udp_camera_names:
            if COMPRESS:
                _ = image.create_dataset(cam_name, (data_tick, padded_size), 
                                         dtype='uint8',
                                         chunks=(1, padded_size), )
            else:
                _ = image.create_dataset(cam_name, (data_tick, 360, 1280, 3), 
                                         dtype='uint8',
                                         chunks=(1, 360, 1280, 3), )
        
        for tracker_name in tracker_names:
            _ = tracker_pose.create_dataset(tracker_name, (data_tick, 7))
            

        for name, array in data_dict.items():
            root[name][...] = array

        if COMPRESS:
            _ = root.create_dataset('compressed_image_len', (len(usb_camera_names+udp_camera_names), data_tick))
            root['/compressed_image_len'][...] = compressed_image_len

    print(f'Saving: {time.time() - t0:.1f} secs \n')
    print("Exiting...")
        
def signal_handler(sig, frame):
    """Handle Ctrl+C and propagate to child processes."""
    print("[INFO] Ctrl+C detected. Exiting...")
    os._exit(0)  # Forcefully terminate all processes
    
def get_user_input():
    global stop_record_flag, stop_user_input_thread_flag, viveUTracker, usb_cam
    print("Press 'q' to stop data recording.")
    while not stop_user_input_thread_flag:
        user_input = input()
        if user_input.lower() == 'q':
            stop_record_flag = True
            viveUTracker.thread_stop()
            usb_cam.stop()
            break

def get_auto_index(dataset_dir, dataset_name_prefix = '', data_suffix = 'hdf5'):
    max_idx = 1000
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    for i in range(max_idx+1):
        if not os.path.isfile(os.path.join(dataset_dir, f'{dataset_name_prefix}episode_{i}.{data_suffix}')):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='Task name.', default='UMI_test', required=False)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', default=None, required=False)
    main(vars(parser.parse_args()))

