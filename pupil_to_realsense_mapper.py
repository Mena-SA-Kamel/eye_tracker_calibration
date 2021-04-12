import zmq
import msgpack
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pyrealsense2 as rs
from PIL import Image

def fetch_realsense_frame(pipeline, align):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
    # Hole filling to get a clean depth image
    hole_filling = rs.hole_filling_filter()
    aligned_depth_frame = hole_filling.process(aligned_depth_frame)
    return [aligned_depth_frame, color_frame, intrinsics]

ctx = zmq.Context()
# The REQ talks to Pupil remote and receives the session unique IPC SUB PORT
pupil_remote = ctx.socket(zmq.REQ)
ip = 'localhost'  # If you talk to a different machine use its IP.
port = 50020  # The port defaults to 50020. Set in Pupil Capture GUI.
pupil_remote.connect(f'tcp://{ip}:{port}')
# Request 'SUB_PORT' for reading data
pupil_remote.send_string('SUB_PORT')

sub_port = pupil_remote.recv_string()
# Request 'PUB_PORT' for writing data
pupil_remote.send_string('PUB_PORT')
pub_port = pupil_remote.recv_string()
subscriber = ctx.socket(zmq.SUB)
subscriber.connect(f'tcp://{ip}:{sub_port}')
subscriber.subscribe('gaze.')  # receive all gaze messages


# Defining variables for Intel RealSense Feed
image_width = 640
image_height = 480
fps = 15
# Create an Intel RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, image_width, image_height, rs.format.z16, fps)
config.enable_stream(rs.stream.color, image_width, image_height, rs.format.rgb8, fps)
config.enable_stream(rs.stream.accel)
config.enable_stream(rs.stream.gyro)
profile = pipeline.start(config)
# Create an align object: rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)
colorizer = rs.colorizer()

# Streaming loop
for i in list(range(20)):
    frames = pipeline.wait_for_frames()
# cv2.namedWindow('Eye Tracker Calibration', cv2.WINDOW_AUTOSIZE)

M_t = np.array([[ 0.99642809, -0.01998785, -0.08204601, -0.022062  ],
                [ 0.01818824,  0.9995786 , -0.02262325, -0.02734249],
                [ 0.08246363,  0.02105017,  0.99637174,  0.00180171]])
invertible_M_t = np.concatenate([M_t, np.array([0,0,0,1]).reshape(1,4)], axis=0)
M_t_inverse = np.linalg.inv(invertible_M_t)

frame_counter = 0
try:
    while True:
        # Decoding the messages from the Pupil eye tracker
        topic, payload = subscriber.recv_multipart()
        message = msgpack.loads(payload)
        normalized_gaze = message[b'norm_pos']
        gaze_point_3d = message[b'gaze_point_3d'] # [x, y, z]
        confidence = message[b'confidence']

        if frame_counter == 0:
            [aligned_depth_frame, color_frame, intrinsics] = fetch_realsense_frame(pipeline, align)
        if not aligned_depth_frame or not color_frame:
            continue
        frame_counter += 1
        color_image = np.asanyarray(color_frame.get_data())
        realsense_world_view = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        gaze_point_3d.append(1)
        gi = np.array(gaze_point_3d).reshape(4, 1)
        gaze_points_realsense_world = np.dot(M_t_inverse, gi) # 3D points. Need to project to the image plance using
                                                              # realsense intrinsincs
        realsense_intrinsics_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                                      [0, intrinsics.fy, intrinsics.ppy],
                                      [0, 0, 1]])

        gaze_points_realsense_image = np.dot(realsense_intrinsics_matrix, gaze_points_realsense_world[:3])
        gaze_points_realsense_image = gaze_points_realsense_image / gaze_points_realsense_image[-1]
        gaze_x_realsense, gaze_y_realsense, _ = gaze_points_realsense_image.squeeze().astype('uint16')
        print(gaze_x_realsense, gaze_y_realsense)
        realsense_world_view = cv2.circle(realsense_world_view, (gaze_x_realsense, gaze_y_realsense), 20, (255, 0, 255), 3)
        realsense_world_view = cv2.circle(realsense_world_view, (gaze_x_realsense, gaze_y_realsense), 2, (255, 0, 255), 2)
        cv2.imshow('Eye Tracker Calibration', realsense_world_view)
        cv2.waitKey(1)


except KeyboardInterrupt:
    pass
finally:
    cv2.destroyAllWindows()
