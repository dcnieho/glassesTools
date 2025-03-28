# this script requires the following python packages:
# pip install projectaria-tools moviepy==1.0.3 pycolmap ffmpeg-binaries pandas
import os
import datetime
import json
import numpy as np
import pandas as pd
import cv2
import pycolmap

from projectaria_tools.core import calibration, data_provider, mps, sophus
from projectaria_tools.core.sensor_data import TimeDomain
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.mps import MpsDataPathsProvider
from projectaria_tools.utils.vrs_to_mp4_utils import convert_vrs_to_mp4

# ensure ffmpeg binaries are on path
import ffmpeg
ffmpeg.add_to_path()


vrs_file        = r"C:\Users\huml-dkn\Aria\f8f41ed1-39dd-4bcb-92e4-cc62d5670080.vrs"
mps_folder      = None   # if not provided, folder is guessed based on default path relative to vrs file as created by Aria Studio
output_folder   = None   # if not provided, folder is made in the directory containing the vrs file


print(f"Creating data provider from {vrs_file}")
provider = data_provider.create_vrs_data_provider(vrs_file)
if not provider:
    print("Invalid vrs data provider")

print('The following streams are present in this recording:')
rgb_id = StreamId("214-1")
et_id  = StreamId("211-1")
have_rgb, have_et = False, False
streams = provider.get_all_streams()
for stream_id in streams:
    print(f"  stream_id: {stream_id} [{provider.get_label_from_stream_id(stream_id)}]")
    if stream_id==rgb_id:
        have_rgb = True
    elif stream_id==et_id:
        have_et = True
if not have_rgb:
    raise RuntimeError("No RGB camera stream found in recording, cannot use this recording.")
if not have_et:
    raise RuntimeError("No eye tracking camera stream found in recording, cannot use this recording.")

rgb_start_time = provider.get_first_time_ns(rgb_id, TimeDomain.DEVICE_TIME)
rgb_end_time   = provider.get_last_time_ns(rgb_id, TimeDomain.DEVICE_TIME)
print(f'RGB camera stream is {(rgb_end_time-rgb_start_time)/1000/1000/1000:.3f} s long ({rgb_start_time}--{rgb_end_time} ns)')
et_start_time = provider.get_first_time_ns(et_id, TimeDomain.DEVICE_TIME)
et_end_time   = provider.get_last_time_ns(et_id, TimeDomain.DEVICE_TIME)
print(f'ET camera stream is {(et_end_time-et_start_time)/1000/1000/1000:.3f} s long ({et_start_time}--{et_end_time} ns)')
print(f'offset between RGB and ET camera at onset, offset is {(rgb_start_time-et_start_time)/1000/1000:.3f},{(rgb_end_time-et_end_time)/1000/1000:.3f} ms')

def image_config(config):
    print(f"  device_type {config.device_type}")
    print(f"  device_version {config.device_version}")
    print(f"  device_serial {config.device_serial}")
    print(f"  sensor_serial {config.sensor_serial}")
    print(f"  nominal_rate_hz {config.nominal_rate_hz}")
    print(f"  image_width {config.image_width}")
    print(f"  image_height {config.image_height}")
    print(f"  pixel_format {config.pixel_format}")
    print(f"  gamma_factor {config.gamma_factor}")
rgb_config = provider.get_image_configuration(rgb_id)
print('Information about camera image stream:')
image_config(rgb_config)
et_config = provider.get_image_configuration(et_id)
print('Information about eye tracking image stream:')
image_config(et_config)

# extract mp4 from vrs file
vrs_path_components = os.path.split(vrs_file)
vrs_file_stem = os.path.splitext(vrs_path_components[-1])[0]
if output_folder is None:
    output_folder = os.path.join(*vrs_path_components[0:-1],f'export_{vrs_file_stem}')
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)
convert_vrs_to_mp4(vrs_file,os.path.join(output_folder,'worldCamera.mp4'))

# get MPS output
if mps_folder is None:
    mps_folder = os.path.join(*vrs_path_components[0:-1],f'mps_{vrs_file_stem}_vrs')
mps_data_paths = MpsDataPathsProvider(mps_folder)
if not (eye_gaze_path := mps_data_paths.get_data_paths().eyegaze.personalized_eyegaze):
    eye_gaze_path = mps_data_paths.get_data_paths().eyegaze.general_eyegaze
gaze_cpf = mps.read_eyegaze(eye_gaze_path)

# this flag should be set to true. convert_vrs_to_mp4() above turns the video upright. We need to make sure
# we export both gaze and a camera calibration that take this rotation into account.
make_upright = True

# get device geometry and camera calibration, needed to project gaze onto the RGB camera feed
device_calibration = provider.get_device_calibration()
# camera calibration
rgb_camera_calibration = device_calibration.get_camera_calib(provider.get_label_from_stream_id(rgb_id))
# get transformation from CPF to rgb camera frame, as per projectaria_tools.core.mps.utils.get_gaze_vector_reprojection
T_device_CPF = device_calibration.get_transform_device_cpf()
T_device_rgb_camera = device_calibration.get_transform_device_sensor(provider.get_label_from_stream_id(rgb_id),True)
if make_upright:
    T_device_rgb_camera = T_device_rgb_camera @ sophus.SE3.from_matrix(np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
T_rgb_camera_cpf = T_device_rgb_camera.inverse() @ T_device_CPF

# get camera calibration in colmap format
if rgb_camera_calibration.get_model_name()!=calibration.CameraModelType.FISHEYE624:
    raise ValueError(f'RGB camera should be calibrated using a Fisheye624 model, anything else ({rgb_camera_calibration.get_model_name()}) is not supported')
if make_upright:
    rgb_camera_calibration = calibration.rotate_camera_calib_cw90deg(rgb_camera_calibration)
cam_cal = pycolmap.Camera.create(0, pycolmap.CameraModelId.RAD_TAN_THIN_PRISM_FISHEYE, rgb_camera_calibration.get_focal_lengths()[0], *rgb_camera_calibration.get_image_size())
cal_params = np.zeros((cam_cal.extra_params_idxs()[-1]+1,))
cal_params[cam_cal.focal_length_idxs()]     = rgb_camera_calibration.get_focal_lengths()
cal_params[cam_cal.principal_point_idxs()]  = rgb_camera_calibration.get_principal_point()
# index the output of rgb_camera_calibration.get_projection_params() by means of number of expected parameters
# as i want to be robust to there be a single or two focal lengths listed in the output of get_projection_params()
cal_params[cam_cal.extra_params_idxs()]     = rgb_camera_calibration.get_projection_params()[-len(cam_cal.extra_params_idxs()):]
cam_cal.params = cal_params # set the parameters for the colmap camera object

# store camera calibration and extrinsics (transformation from CPF to RGB camera) to XML file
camera_info = {
    'resolution': rgb_camera_calibration.get_image_size(),
    'position': T_rgb_camera_cpf.to_quat_and_translation()[0][4:]*1000.,    # m -> mm
    'rotation': T_rgb_camera_cpf.to_matrix()[:3,:3],
    'colmap_camera': cam_cal.todict()
}
# fix up some values that cv2.FileStorage can't handle
camera_info['colmap_camera']['model']                   =     camera_info['colmap_camera']['model'].name
camera_info['colmap_camera']['has_prior_focal_length']  = int(camera_info['colmap_camera']['has_prior_focal_length'])
# store to file
fs = cv2.FileStorage(os.path.join(output_folder, 'calibration.xml'), cv2.FILE_STORAGE_WRITE)
for key,value in camera_info.items():
    if isinstance(value,dict):
        fs.startWriteStruct('colmap_camera', cv2.FileNode_MAP)
        for dkey,dvalue in value.items():
            fs.write(name=dkey,val=dvalue)
        fs.endWriteStruct()
    else:
        fs.write(name=key,val=value)
fs.release()

# get gaze data to export
samples: list[np.ndarray] = []
for i,s in enumerate(gaze_cpf):
    # get 3D binocular gaze point
    binocular_gaze_point_cpf = mps.get_eyegaze_point_at_depth(s.yaw, s.pitch, s.depth or 1.0) # If depth available use it, else fall back to 1 meter depth along the EyeGaze ray
    # get gaze position on camera image
    binocular_gaze_point_rgb_camera = T_rgb_camera_cpf @ binocular_gaze_point_cpf
    gaze_position_rgb_camera = cam_cal.img_from_cam(np.reshape(binocular_gaze_point_rgb_camera,(1,3)))
    # get gaze vectors
    gaze_vectors = mps.get_gaze_vectors(s.vergence.left_yaw, s.vergence.right_yaw, s.pitch)
    # get origins of gaze vectors (convert from m to mm)
    gaze_ori = np.array([s.vergence.tx_left_eye, s.vergence.ty_left_eye, s.vergence.tz_left_eye, s.vergence.tx_right_eye, s.vergence.ty_right_eye, s.vergence.tz_right_eye])*1000.
    # get timestamp, in relative video time
    ts = s.tracking_timestamp/datetime.timedelta(microseconds=1)-int(rgb_start_time/1000)
    # store (convert binocular gaze point from m to mm)
    samples.append(np.concatenate(([ts],gaze_position_rgb_camera.flatten(),binocular_gaze_point_cpf*1000.,*gaze_vectors,gaze_ori)))

# turn into data frame, save
gaze_df = pd.DataFrame(samples,columns=['timestamp','gaze_pos_vid_x','gaze_pos_vid_y','gaze_pos_3d_x','gaze_pos_3d_y','gaze_pos_3d_z','gaze_dir_left_x','gaze_dir_left_y','gaze_dir_left_z','gaze_dir_right_x','gaze_dir_right_y','gaze_dir_right_z','gaze_ori_left_x','gaze_ori_left_y','gaze_ori_left_z','gaze_ori_right_x','gaze_ori_right_y','gaze_ori_right_z'])
gaze_df.to_csv(os.path.join(output_folder, 'gaze.tsv'), sep='\t', float_format="%.8f", index=False, na_rep='nan')

# finally, get meta data
md = provider.get_metadata()
metadata = {}
metadata['start_time'] = md.start_time_epoch_sec
metadata['glasses_serial'] = md.device_serial
metadata['duration'] = int((max(et_start_time, et_end_time)-min(rgb_start_time,rgb_end_time))/1000.)
metadata['scene_camera_serial'] = rgb_config.sensor_serial
metadata['name'] = vrs_file_stem
# can get some extra metadata from vrs's json file, if present
vrs_json_file = vrs_file+'.json'
if os.path.isfile(vrs_json_file):
    with open(vrs_json_file,'r') as f:
        md2 = json.load(f)
    if 'firmware_version' in md2:
        metadata['firmware_version'] = md2['firmware_version']
    if 'companion_version' in md2 and md2['companion_version']:
        metadata['recording_software_version'] = md2['companion_version']

with open(os.path.join(output_folder, 'metadata.json'),'w') as f:
    json.dump(metadata, f)