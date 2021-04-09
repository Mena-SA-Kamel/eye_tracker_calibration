import zmq
from msgpack import unpackb, packb
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pyrealsense2 as rs
from PIL import Image


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

def notify(notification):
    """Sends ``notification`` to Pupil Remote"""
    topic = 'notify.' + notification['subject']
    payload = packb(notification, use_bin_type=True)
    req.send_string(topic, flags=zmq.SNDMORE)
    req.send(payload)
    return req.recv_string()

def recv_from_sub():
    '''Recv a message with topic, payload.
    Topic is a utf-8 encoded string. Returned as unicode object.
    Payload is a msgpack serialized dict. Returned as a python dict.
    Any addional message frames will be added as a list
    in the payload dict with key: '__raw_data__' .
    '''
    topic = sub.recv_string()
    payload = unpackb(sub.recv(), encoding='utf-8')
    extra_frames = []
    while sub.get(zmq.RCVMORE):
        extra_frames.append(sub.recv())
    if extra_frames:
        payload['__raw_data__'] = extra_frames
    return topic, payload


def get_world_frame_from_pupil():
    eyetracker_image = None
    try:
        while True:
            topic, msg = recv_from_sub()
            if topic == 'frame.world':
                eyetracker_image = np.frombuffer(msg['__raw_data__'][0], dtype=np.uint8).reshape(msg['height'],
                                                                                             msg['width'], 3)
                break
    except KeyboardInterrupt:
        pass
    return eyetracker_image

def get_realsense_frame():
    # Getting the world frame from Intel RealSense Camera
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
    realsense_world_view = np.zeros((image_height, image_width, 3))
    # Streaming loop
    for i in list(range(20)):
        frames = pipeline.wait_for_frames()
    while True:
        try:
            frames = pipeline.wait_for_frames()
            if np.mean(realsense_world_view) == 0:
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
                realsense_world_view = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
                break
        except:
            continue
    return (realsense_world_view, intrinsics, aligned_depth_frame, pipeline)

def plot_frames(frame1, frame2):
    fig, axs = plt.subplots(2, figsize=(7, 7))
    axs[0].imshow(frame1.astype(np.uint8))
    axs[0].set_title('Frame 1')
    axs[1].imshow(frame2.astype(np.uint8))
    axs[1].set_title('Frame 2')
    plt.show(block=False)


# Getting the world frame from Pupil eye tracker
context = zmq.Context()
# open a req port to talk to pupil
addr = '127.0.0.1'  # remote ip or localhost
req_port = "50020"  # same as in the pupil remote gui
req = context.socket(zmq.REQ)
req.connect("tcp://{}:{}".format(addr, req_port))
# ask for the sub port
req.send_string('SUB_PORT')
sub_port = req.recv_string()
# Start frame publisher with format BGR
notify({'subject': 'start_plugin', 'name': 'Frame_Publisher', 'args': {'format': 'bgr'}})
# open a sub port to listen to pupil
sub = context.socket(zmq.SUB)
sub.connect("tcp://{}:{}".format(addr, sub_port))
# set subscriptions to topics
# recv just pupil/gaze/notifications
sub.setsockopt_string(zmq.SUBSCRIBE, 'frame.')

pupil_world_view = get_world_frame_from_pupil()
realsense_world_view, realsense_intrinsics, aligned_depth_frame, pipeline = get_realsense_frame()



im_pupil = Image.fromarray(pupil_world_view)
im_pupil.save("im_pupil.jpeg")

im_realsense = Image.fromarray(realsense_world_view)
im_realsense.save("im_realsense.jpeg")

#
# pupil_world_view = Image.open("im_pupil.jpeg")
# realsense_world_view = Image.open("im_realsense.jpeg")
# pupil_world_view = np.asarray(pupil_world_view)
# realsense_world_view = np.asarray(realsense_world_view)


cv2.namedWindow('Eye Tracker Calibration', cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback('Eye Tracker Calibration', onMouse)
font = cv2.FONT_HERSHEY_SIMPLEX
prev_mouse_location = [0, 0]
num_points = 5
labelled_images = []

images_to_annotate = [pupil_world_view, realsense_world_view]
image_labels = ['Pupil Frame', 'RealSense Frame']
points = np.zeros((2, num_points, 2), dtype='uint16')

for i, image in enumerate(images_to_annotate):
    pixel_locations = []
    while len(pixel_locations) < num_points:
        if [mouseX, mouseY] != prev_mouse_location:
            # Getting the XYZ coordinates of the grasping box center <x, y>
            pixel_locations.append([mouseX, mouseY])
            prev_mouse_location = [mouseX, mouseY]
        for j, pixel_location in enumerate(pixel_locations):
            x, y = pixel_location
            image = cv2.circle(image, (x, y), 20, (255, 0, 0), 1)
            image = cv2.circle(image, (x, y), 2, (255, 0, 0), 2)
            cv2.putText(image, str(j), (x + 4, y), font, 0.5, (0, 0, 255), thickness=2)
        cv2.imshow('Eye Tracker Calibration', image)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
    points[i] = np.array(pixel_locations, dtype='uint16')

    labelled_images.append(image)

pupil_points = points[0]
realsense_image_points = points[1]
realsense_world_points = np.zeros((num_points, 3))
for k, image_frame_points in enumerate(realsense_image_points):
    realsense_world_points[k] = de_project_point(realsense_intrinsics, aligned_depth_frame, image_frame_points)

np.savetxt('world_locations.txt', realsense_world_points)
np.savetxt('gaze_points.txt', pupil_points)
pipeline.stop()