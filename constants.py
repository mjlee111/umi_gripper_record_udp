import pathlib
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
print(CURRENT_DIR)
os.makedirs(CURRENT_DIR + '/data', exist_ok=True)

DATA_DIR = CURRENT_DIR + '/data'
TASK_CONFIGS = {
    'UMI_test':{
        'dataset_dir': DATA_DIR + '/UMI_test',
        'episode_len': 1200,
        'train_ratio': 0.99,
        'usb_camera_names': ['head_camera'],
        'udp_camera_names': ['lhand_camera'],
        'ip_dict': {
            'lhand_camera': ('0.0.0.0', 11001), 
        },
        'tracker_names': ['left_wrist', 'chest'],
        'name_filter': lambda n: 'sort_only' in n,
        'visualize': True,
    },
    'UMI_RECORD':{
        'dataset_dir': DATA_DIR + '/UMI_test',
        'episode_len': 1200,
        'train_ratio': 0.99,
        'usb_camera_names': ['head_camera'],
        'udp_camera_names': ['lhand_camera', 'rhand_camera'],
        'ip_dict': {
            'lhand_camera': ('0.0.0.0', 11001), 
            'rhand_camera': ('0.0.0.0', 11002),
        },
        'tracker_names': ['left_wrist', 'chest', 'right_wrist'],
        'name_filter': lambda n: 'sort_only' in n,
        'visualize': True,
    }
    
}

HZ = 20
DT = 1/HZ

XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/assets/'