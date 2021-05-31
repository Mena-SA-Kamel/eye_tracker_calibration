import zmq
import msgpack
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pyrealsense2 as rs
from PIL import Image
import threading

def fetch_realsense_frame(pipeline, align, aligned_depth_frame, color_frame):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame_object = aligned_frames.get_depth_frame()
    color_frame[0] = aligned_frames.get_color_frame()
    # Hole filling to get a clean depth image
    hole_filling = rs.hole_filling_filter()
    aligned_depth_frame[0] = hole_filling.process(aligned_depth_frame_object)

def fetch_gaze_vector(subscriber, avg_gaze):
    while True:
        topic, payload = subscriber.recv_multipart()
        message = msgpack.loads(payload)
        gaze_point_3d = message[b'gaze_point_3d']
        avg_gaze[:] = gaze_point_3d

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

M_t = np.array([[0.98323309, -0.03063463,  0.17976155, -0.05710686],
                [ 0.03361795,  0.9993426,  -0.01357236, -0.02003963],
                [-0.17922759,  0.01938801,  0.98361658,  0.01666406]])

tvec = M_t[:,-1]
rvec, jacobian = cv2.Rodrigues(M_t[:,:3])


invertible_M_t = np.concatenate([M_t, np.array([0,0,0,1]).reshape(1,4)], axis=0)
M_t_inverse = np.linalg.inv(invertible_M_t)

realsense_intrinsics_matrix = np.array([[609.87304688,   0.        , 332.6171875 ],
                                        [  0.        , 608.84387207, 248.34165955],
                                        [  0.        ,   0.        ,   1.        ]])

frame_counter = 0
avg_gaze = [0,0,0]
aligned_depth_frame = [None]
color_frame = [None]

# Thread runs infinetly
t1 = threading.Thread(target=fetch_gaze_vector, args=(subscriber, avg_gaze))
t1.start()

try:
    while True:
        # Getting an RGB and Depth frame from RealSense Camera - THREAD 2
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame_object = aligned_frames.get_depth_frame()
        color_frame[0] = aligned_frames.get_color_frame()
        # Hole filling to get a clean depth image
        hole_filling = rs.hole_filling_filter()
        aligned_depth_frame[0] = hole_filling.process(aligned_depth_frame_object)
        # t2 = threading.Thread(target=fetch_realsense_frame, args=(pipeline, align, aligned_depth_frame, color_frame))
        # t2.start()
        # t2.join() # Wait until the frame is capture from the RealSense camera before moving on
        if not aligned_depth_frame[0] or not color_frame[0]:
            continue
        frame_counter += 1
        color_image = np.asanyarray(color_frame[0].get_data())
        realsense_world_view = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        gaze_points = np.array(avg_gaze).reshape(-1, 1, 3)
        gaze_points_realsense_image, jacobian = cv2.projectPoints(gaze_points, rvec, tvec, realsense_intrinsics_matrix, None)
        gaze_x_realsense, gaze_y_realsense = gaze_points_realsense_image.squeeze().astype('uint16')

        # gaze_vector = list(avg_gaze)
        # gaze_vector.append(1)
        # gi = np.array(gaze_vector).reshape(4, 1)
        # gaze_points_realsense_world = np.dot(invertible_M_t, gi) # 3D points. Need to project to the image plane using
        #                                                          # realsense intrinsincs
        # gaze_points_realsense_image = np.dot(realsense_intrinsics_matrix, gaze_points_realsense_world[:3])
        # gaze_points_realsense_image = gaze_points_realsense_image / gaze_points_realsense_image[-1]

        # gaze_x_realsense, gaze_y_realsense, _ = gaze_points_realsense_image.squeeze().astype('uint16')
        realsense_world_view = cv2.circle(realsense_world_view, (gaze_x_realsense, gaze_y_realsense), 20, (255, 0, 255), 3)
        realsense_world_view = cv2.circle(realsense_world_view, (gaze_x_realsense, gaze_y_realsense), 2, (255, 0, 255), 2)
        cv2.imshow('Eye Tracker Calibration', realsense_world_view)
        cv2.waitKey(1)
except KeyboardInterrupt:
    pass
finally:
    cv2.destroyAllWindows()
