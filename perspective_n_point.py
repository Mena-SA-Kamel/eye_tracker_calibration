import cv2
import numpy as np
import matplotlib.pyplot as plt

Pi_list = np.loadtxt("world_locations.txt")
# gi_list = np.loadtxt("mean_gaze_vectors.txt")
# wi_list = np.loadtxt("mean_gaze_vectors.txt")[:, -1]
gi_list = np.loadtxt("gaze_points.txt")

num_points = 5
# Reshaping Pi_list into a 6x1 3-channel matrix
object_points = np.array(Pi_list, dtype=np.float32).reshape(-1,1,3)

# Perspective projecting all gaze vectors and reshaping gi_list into a 6x1 2-channel matrix
# image_points = cv2.convertPointsFromHomogeneous(np.float32(gi_list)).reshape(num_points,1,2)
image_points = np.float32(gi_list).reshape(num_points,1,2)

# # Defining camera_matrix as an identity matrix of size 3x3
camera_matrix = np.array([[794.3311439869655, 0.0, 633.0104437728625],
                [0.0, 793.5290139393004, 397.36927353414865],
                [0.0, 0.0, 1.0]])

# Defining zero distortion coefficients
dist_coef = np.array([ -0.3758628065070806,
                    0.1643326166951343,
                    0.00012182540692089567,
                    0.00013422608638039466,
                    0.03343691733865076,
                    0.08235235770849726,
                    -0.08225804883227375,
                    0.14463365333602152])

# # Defining camera_matrix as an identity matrix of size 3x3
# camera_matrix = np.identity(3, dtype=np.float32)
#
# # Defining zero distortion coefficients
# dist_coef = np.zeros(4)

# Cross calibration with EPnP
ret, rvec, tvec = cv2.solvePnP(object_points, image_points,
            camera_matrix, dist_coef, None, None, False, cv2.SOLVEPNP_EPNP)

# Using cv2.Rodrigues() to convert rvec from a rotation vector to a rotation matrix
rvec_matrix, jacobian = cv2.Rodrigues(rvec)

# Constructing the Mt matrix (camera extrinsics)
Mt =  np.concatenate((rvec_matrix, tvec),1)
np.set_printoptions(suppress=True)
print ("\n###### Mt Matrix ######")
print(Mt)

# Verifying Mt matrix by projecting the 3D points from world frame -> camera frame -> image plane
reconstructed_image_points, jacobian = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, None)
net_deviation = abs(reconstructed_image_points - image_points)

# Visualize the differences between original 2D points and reconstructed points
image_points_new = image_points.reshape(num_points,2)
x = image_points_new[:,0]
y = image_points_new[:,1]
reconstructed_image_points_new = reconstructed_image_points.reshape(num_points,2)
x_recon = reconstructed_image_points_new[:,0]
y_recon = reconstructed_image_points_new[:,1]
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(x, y, c='b', marker="s", label='Original Virtual Points')
ax1.scatter(x_recon,y_recon, c='r', marker="o", label='Reconstructed Virtual Points')
plt.legend(loc='upper right');
plt.xlabel('x')
plt.ylabel('y')
plt.title('Projection of Pi unto virtual plane')
plt.savefig('Deltas.png')
plt.show()

import code; code.interact(local=dict(globals(), **locals()))
