import time
from pyquaternion import Quaternion
import threading
import open3d as o3d
import numpy as np

class VIVEOpen3DVisualizer:
    def __init__(self):
        # Initialize Open3D visualizer
        
        self.vis_obj_offset_rotm = np.array([[-1, 0, 0],[0, 0, 1],[0, 1, 0]])
        # Create three smaller 3D cubes
        # self.cube_red = o3d.geometry.TriangleMesh.create_box(width=0.2, height=0.03, depth=0.1)
        self.cube_red = o3d.io.read_triangle_mesh('C:/Users/dgyo3/Desktop/dg/ROBROS_UMI/asset/Ultimate Tracker_1115.obj', enable_post_processing = True, print_progress = True)
        if not self.cube_red.is_empty():
            bbox = self.cube_red.get_axis_aligned_bounding_box()
            print("Bounding Box Dimensions:", bbox.get_extent())
            print("Successfully loaded .obj file")
        else:
            print("Failed to load .obj file")
        self.cube_red.scale(0.001, self.cube_red.get_center())
        self.cube_red.rotate(self.vis_obj_offset_rotm)
        self.cube_red.create_coordinate_frame(size = 1000.0)
        self.cube_red.paint_uniform_color([1, 0, 0])  # Red color
        
        # self.cube_blue = o3d.geometry.TriangleMesh.create_box(width=0.2, height=0.03, depth=0.1)
        self.cube_blue = o3d.io.read_triangle_mesh('C:/Users/dgyo3/Desktop/dg/ROBROS_UMI/asset/Ultimate Tracker_1115.obj', enable_post_processing = True, print_progress = True)
        self.cube_blue.scale(0.001, self.cube_blue.get_center())
        self.cube_blue.rotate(self.vis_obj_offset_rotm)
        self.cube_blue.paint_uniform_color([0, 0, 1])  # Blue color
        self.cube_blue.create_coordinate_frame(size = 1000.0)
        
        # self.cube_green = o3d.geometry.TriangleMesh.create_box(width=0.2, height=0.03, depth=0.1)
        self.cube_green = o3d.io.read_triangle_mesh('C:/Users/dgyo3/Desktop/dg/ROBROS_UMI/asset/Ultimate Tracker_1115.obj', enable_post_processing = True, print_progress = True)
        self.cube_green.scale(0.001, self.cube_green.get_center())
        self.cube_green.rotate(self.vis_obj_offset_rotm)
        self.cube_green.create_coordinate_frame(size = 1000.0)
        self.cube_green.paint_uniform_color([0, 1, 0])  # Green color

        # Add the cubes to the visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        
        self.vis.add_geometry(self.cube_red)
        self.vis.add_geometry(self.cube_blue)
        self.vis.add_geometry(self.cube_green)
        # o3d.visualization.draw_geometries([self.cube_blue, self.cube_green])
        # o3d.visualization.draw_geometries(self.cube_blue)
        # o3d.visualization.draw_geometries(self.cube_green)
        
        self.initial_transformations = {
            0: np.eye(4),
            1: np.eye(4),
            2: np.eye(4)
        }
        self.cube_centers = {
            0: self.cube_red.get_center(),
            1: self.cube_blue.get_center(),
            2: self.cube_green.get_center()
        }

    def set_pose_first(self, translation, quaternion, cube_id):
        # # Set initial frame offset
        # cube.translate(-cube_center)
        # cube.transform(self.vis_obj_offset_transformations)
        # cube.translate(cube_center)
        
        # Set initial world frame using translation and quaternion
        initial_translation = translation
        initial_quaternion = Quaternion(quaternion)

        # Create initial transformation matrix
        initial_transformation = np.eye(4)
        initial_transformation[:3, :3] = initial_quaternion.rotation_matrix
        initial_transformation[:3, 3] = initial_translation

        # Get the corresponding cube and its center
        cube = self._get_cube(cube_id)
        cube_center = self.cube_centers[cube_id]

        # Center the cube at the origin before applying transformations
        cube.translate(-cube_center)
        cube.transform(initial_transformation)
        cube.translate(cube_center)
        
        # Store the initial transformation
        self.initial_transformations[cube_id] = initial_transformation

        # Set the view control to ensure the cube is within view
        view_control = self.vis.get_view_control()
        view_control.set_zoom(5.0)  # Adjust zoom level as needed
        view_control.set_lookat(cube_center)
        view_control.set_front([0, 0, -1])
        view_control.set_up([0, 1, 0])  # Corrected up vector

        # Adjust clipping planes to ensure the cube doesn't disappear
        camera_params = view_control.convert_to_pinhole_camera_parameters()
        view_control.convert_from_pinhole_camera_parameters(camera_params)
        view_control.change_field_of_view(step=20)  # Increase FOV for better visibility
        # camera_params = view_control.convert_to_pinhole_camera_parameters()
        # camera_params.extrinsic[2, 3] = -5
        # camera_params.intrinsic.set_intrinsics(1920, 1080, 1000, 1000, 960, 540)  # Adjust FOV if needed
        # view_control.convert_from_pinhole_camera_parameters(camera_params)
        view_control.set_constant_z_near(-100000.0)
        view_control.set_constant_z_far(100000.0)  # Adjust far plane to a large value

    def set_pose(self, translation, quaternion, cube_id):
        # Get the corresponding cube and its center
        cube = self._get_cube(cube_id)
        cube_center = self.cube_centers[cube_id]

        # Reset to initial pose
        cube.translate(-cube_center)
        cube.transform(np.linalg.inv(self.initial_transformations[cube_id]))
        cube.translate(cube_center)

        # Update pose in the same world frame
        new_quaternion = Quaternion(quaternion)
        new_transformation = np.eye(4)
        new_transformation[:3, :3] = new_quaternion.rotation_matrix
        new_transformation[:3, 3] = translation

        # Center the cube at the origin before applying transformations
        cube.translate(-cube_center)
        cube.transform(new_transformation)
        cube.translate(cube_center)
        
        # Store the new transformation
        self.initial_transformations[cube_id] = new_transformation

        # Update visualizer
        self.vis.update_geometry(cube)
        self.vis.poll_events()
        self.vis.update_renderer()

    def _get_cube(self, cube_id):
        if cube_id == 0:
            return self.cube_red
        elif cube_id == 1:
            return self.cube_blue
        elif cube_id == 2:
            return self.cube_green
        else:
            raise ValueError("Invalid cube_id")
            
    def start_visualizer(self):
        threading.Thread(target=self.run, daemon=True).start()

# Example usage:
# visualizer = Open3DVisualizer()
# visualizer.start_visualizer()
# visualizer.set_pose_first([0, 0, 0], [1, 0, 0, 0], 0)
# visualizer.set_pose_first([0.5, 0.5, 0], [1, 0, 0, 0], 1)
# visualizer.set_pose_first([1, 1, 1], [1, 0, 0, 0], 2)
# visualizer.set_pose([0.1, 0.1, 0.1], [0.707, 0, 0.707, 0], 0)
# visualizer.set_pose([0.6, 0.6, 0.1], [0.707, 0, 0.707, 0], 1)
# visualizer.set_pose([1.1, 1.1, 1.1], [0.707, 0, 0.707, 0], 2)
