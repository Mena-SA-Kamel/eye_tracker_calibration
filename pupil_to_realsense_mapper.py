import zmq
import msgpack
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pyrealsense2 as rs
from PIL import Image
import threading, queue

def fetch_realsense_frame(pipeline, align, aligned_depth_frame, color_frame):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame_object = aligned_frames.get_depth_frame()
    color_frame[0] = aligned_frames.get_color_frame()
    # Hole filling to get a clean depth image
    hole_filling = rs.hole_filling_filter()
    aligned_depth_frame[0] = hole_filling.process(aligned_depth_frame_object)

def fetch_gaze_vector(subscriber, avg_gaze, n):
    while True:
        topic, payload = subscriber.recv_multipart()
        message = msgpack.loads(payload)
        gaze_point_3d = message[b'gaze_point_3d']
        avg_gaze[:] = gaze_point_3d
    # gaze_point_3d = np.array(gaze_point_3d)
    # prev_mean = np.array(avg_gaze)
    # if n[0] == 0:
    #     avg_gaze[:] = list(gaze_point_3d)
    # else:
    #     avg_gaze[:] = list(prev_mean + (gaze_point_3d - prev_mean) / n[0])
    # n[0] += 1

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

M_t = np.array([[  0.99866882,  0.01168762,  0.05023925, -0.13939386],
 [-0.01221465,  0.99987341,  0.01019624, -0.00047303],
 [-0.05011372, -0.01079632,  0.99868516, -0.0069492]])

invertible_M_t = np.concatenate([M_t, np.array([0,0,0,1]).reshape(1,4)], axis=0)
M_t_inverse = np.linalg.inv(invertible_M_t)

realsense_intrinsics_matrix = np.array([[609.87304688,   0.        , 332.6171875 ],
                                        [  0.        , 608.84387207, 248.34165955],
                                        [  0.        ,   0.        ,   1.        ]])

frame_counter = 0
avg_gaze = [0,0,0]
n = [0]
aligned_depth_frame = [None]
color_frame = [None]

# Thread runs infinetly
t1 = threading.Thread(target=fetch_gaze_vector, args=(subscriber, avg_gaze, n))
t1.start()

try:
    while True:
        # Decoding the messages from the Pupil eye tracker - THREAD 1
        # [avg_gaze, n] = fetch_gaze_vector(subscriber, avg_gaze, n)

        # Getting an RGB and Depth frame from RealSense Camera - THREAD 2
        t2 = threading.Thread(target=fetch_realsense_frame, args=(pipeline, align, aligned_depth_frame, color_frame))
        t2.start()
        t2.join() # Wait until the frame is capture from the RealSense camera before moving on
        n[0] = 0
        if not aligned_depth_frame[0] or not color_frame[0]:
            continue
        frame_counter += 1
        color_image = np.asanyarray(color_frame[0].get_data())
        realsense_world_view = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        gaze_vector = list(avg_gaze)
        gaze_vector.append(1)
        gi = np.array(gaze_vector).reshape(4, 1)
        gaze_points_realsense_world = np.dot(invertible_M_t, gi) # 3D points. Need to project to the image plane using
                                                              # realsense intrinsincs
        gaze_points_realsense_image = np.dot(realsense_intrinsics_matrix, gaze_points_realsense_world[:3])
        gaze_points_realsense_image = gaze_points_realsense_image / gaze_points_realsense_image[-1]
        # here x and y values are relative to the bottom left corner, needs to be relative to the top left corner for open cv

        gaze_x_realsense, gaze_y_realsense, _ = gaze_points_realsense_image.squeeze().astype('uint16')
        realsense_world_view = cv2.circle(realsense_world_view, (gaze_x_realsense, gaze_y_realsense), 20, (255, 0, 255), 3)
        realsense_world_view = cv2.circle(realsense_world_view, (gaze_x_realsense, gaze_y_realsense), 2, (255, 0, 255), 2)
        cv2.imshow('Eye Tracker Calibration', realsense_world_view)
        cv2.waitKey(1)


except KeyboardInterrupt:
    pass
finally:
    cv2.destroyAllWindows()
