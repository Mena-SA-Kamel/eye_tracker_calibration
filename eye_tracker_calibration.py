# 'R'  # start recording with auto generated session name
# 'R rec_name'  # start recording named "rec_name"
# 'r'  # stop recording
# 'C'  # start currently selected calibration
# 'c'  # stop currently selected calibration
# 'T 1234.56'  # resets current Pupil time to given timestamp
# 't'  # get current Pupil time; returns a float as string.
# 'v'  # get the Pupil Core software version string
#
# # IPC Backbone communication
# 'PUB_PORT'  # return the current pub port of the IPC Backbone
# 'SUB_PORT'  # return the current sub port of the IPC Backbone
import zmq
import time
import msgpack
import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs
import cv2
import winsound


mouseX, mouseY = [0, 0]

def onMouse(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
       mouseX, mouseY = [x, y]

def de_project_point(intrinsics, depth_frame, point):
    # This function deprojects point from the image plane to the camera frame of reference using camera intrinsic
    # parameters
    x, y = point
    distance_to_point = depth_frame.as_depth_frame().get_distance(x, y)
    de_projected_point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], distance_to_point)
    return de_projected_point

frequency = 2500  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second

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
cv2.namedWindow('Eye Tracker Calibration', cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback('Eye Tracker Calibration', onMouse)

prev_mouse_location = [0, 0]
pixel_locations = []
world_locations = []
mean_gaze_vectors = []
font = cv2.FONT_HERSHEY_SIMPLEX
color_image_to_display = np.zeros((image_height, image_width, 3))
try:
    while True:
        cv2.imshow('Eye Tracker Calibration', color_image_to_display)
        if len(pixel_locations) < 5:
            frames = pipeline.wait_for_frames()
            if np.mean(color_image_to_display) == 0:
                 # Align the depth frame to color frame
                aligned_frames = align.process(frames)
                aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
                color_frame = aligned_frames.get_color_frame()
                intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
                intrinsics_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                                              [0, intrinsics.fy, intrinsics.ppy],
                                              [0, 0, 1]])
                # Hole filling to get a clean depth image
                hole_filling = rs.hole_filling_filter()
                aligned_depth_frame = hole_filling.process(aligned_depth_frame)
                if not aligned_depth_frame or not color_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                color_image_to_display = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

            if [mouseX, mouseY] != prev_mouse_location:
                # Getting the XYZ coordinates of the grasping box center <x, y>
                pixel_locations.append([mouseX, mouseY])
                world_location = de_project_point(intrinsics, aligned_depth_frame, [mouseX, mouseY])
                print ("Pixel Coordinates: ", [mouseX, mouseY], " World Coordinates: ", world_location)
                world_locations.append(world_location)
                prev_mouse_location = [mouseX, mouseY]

            for i, pixel_location in enumerate(pixel_locations):
                x, y = pixel_location
                color_image_to_display = cv2.circle(color_image_to_display, (x, y), 20, (255, 0, 0), 1)
                color_image_to_display = cv2.circle(color_image_to_display, (x, y), 2, (255, 0, 0), 2)
                cv2.putText(color_image_to_display, str(i), (x + 4, y), font, 0.5, (0, 0, 255), thickness=2)
        else:

            # After all points are acquired in the camera frame, loop through each point and wait for a duration of
            # time to average the gaze vector
            eye_gaze_vector = []
            for i, pixel_location in enumerate(pixel_locations):
                winsound.Beep(frequency, duration)
                time.sleep(2) # sleeping until the eye focuses
                x, y = pixel_location
                color_image_to_display = cv2.circle(color_image_to_display, (x, y), 30, (255, 0, 255), 1)
                cv2.imshow('Eye Tracker Calibration', color_image_to_display)

                sample_counter = 0
                while sample_counter < 2000:
                    topic, payload = subscriber.recv_multipart()
                    message = msgpack.loads(payload)
                    normalized_gaze = message[b'norm_pos']
                    gaze_point_3d = message[b'gaze_point_3d']
                    confidence = message[b'confidence']
                    if confidence > 0.0:
                        # eye_gaze_vector.append(gaze_point_3d)
                        eye_gaze_vector.append(normalized_gaze)
                        sample_counter += 1
            eye_gaze_vector = np.array(eye_gaze_vector)
            #eye_gaze_vector_mean = np.mean(eye_gaze_vector, axis = 0)
            #mean_gaze_vectors.append(eye_gaze_vector_mean)
            winsound.Beep(frequency, duration)
            time.sleep(1)
            winsound.Beep(frequency, duration)
            break

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

finally:
    pipeline.stop()

# mean_gaze_vectors -> Array containing the 5 points for calibration in pupil world frame
# mean_gaze_vectors = np.array(mean_gaze_vectors)
# world_locations = np.array(world_locations)
#
# # gaze_points = (mean_gaze_vectors / mean_gaze_vectors[:, -1].reshape(-1, 1))[:, :2]
# gaze_points = mean_gaze_vectors
# np.savetxt('world_locations.txt', world_locations)
# np.savetxt('gaze_points.txt', gaze_points)
# np.savetxt('mean_gaze_vectors.txt', mean_gaze_vectors)

plt.scatter(eye_gaze_vector[:, 0], eye_gaze_vector[:,1]); plt.show(block=False)
import code; code.interact(local=dict(globals(), **locals()))
#
# while counter < 5000:
#     topic, payload = subscriber.recv_multipart()
#     message = msgpack.loads(payload)
#     gaze_point_3d = message[b'gaze_point_3d']
#     print(f"{topic}: {gaze_point_3d}")
#     print("\n")
#     counter+=1
#     if counter %50 == 0:
#         eye_gaze_points.append(gaze_point_3d)
#         print (counter)
# eye_gaze_points = np.array(eye_gaze_points)
#
# plt.plot(eye_gaze_points[:, 0], label="x")
# plt.plot(eye_gaze_points[:, 1], label="y")
# plt.plot(eye_gaze_points[:, 2], label="z")
# plt.legend()
# plt.show()
# import code; code.interact(local=dict(globals(), **locals()))


#
# # start recording
# pupil_remote.send_string('R')
# print(pupil_remote.recv_string())
#
# time.sleep(5)
# pupil_remote.send_string('r')
# print(pupil_remote.recv_string())