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

subscriber.subscribe('fixation')  # receive all gaze messages
eye_gaze_points = []
counter = 0

while counter < 5000:
    import code;

    code.interact(local=dict(globals(), **locals()))
    topic, payload = subscriber.recv_multipart()
    message = msgpack.loads(payload)
    gaze_point_3d = message[b'gaze_point_3d']
    print(f"{topic}: {gaze_point_3d}")
    print("\n")
    counter+=1
    if counter %50 == 0:
        eye_gaze_points.append(gaze_point_3d)
        print (counter)
eye_gaze_points = np.array(eye_gaze_points)

plt.plot(eye_gaze_points[:, 0], label="x")
plt.plot(eye_gaze_points[:, 1], label="y")
plt.plot(eye_gaze_points[:, 2], label="z")
plt.legend()
plt.show()
import code; code.interact(local=dict(globals(), **locals()))
#
# # start recording
# pupil_remote.send_string('R')
# print(pupil_remote.recv_string())
#
# time.sleep(5)
# pupil_remote.send_string('r')
# print(pupil_remote.recv_string())