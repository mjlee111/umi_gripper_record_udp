import pathlib
import os

DATA_DIR = os.path.expanduser('C:/Users/dgyo3/Desktop/dg/IL_data')
TASK_CONFIGS = {
    'UMI_test':{
        'dataset_dir': DATA_DIR + '/UMI_test',
        'episode_len': 1800,
        'train_ratio': 0.99,
        'usb_camera_names': ['head_camera'],
        'udp_camera_dict': {
            'lhand_camera': None, 
        },
        'ip_dict': {
            'lhand_camera': ('0.0.0.0', 5000), 
        },
        'tracker_names': ['left_wrist', 'chest'],
        'name_filter': lambda n: 'sort_only' in n,
    }
}

HZ = 20
DT = 1/HZ

XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/assets/' # note: absolute path