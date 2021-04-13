import cv2
import numpy as np
import matplotlib.pyplot as plt

def solve_pnp(object_points, image_points,camera_matrix, dist_coef, plot=False):
    # Reshaping Pi_list into a 6x1 3-channel matrix
    object_points = object_points.reshape(-1, 1, 3)
    # Perspective projecting all gaze vectors and reshaping gi_list into a 6x1 2-channel matrix
    image_points =image_points.reshape(-1, 1, 2)
    # Cross calibration with EPnP
    ret, rvec, tvec = cv2.solvePnP(object_points, image_points,
                                   camera_matrix, dist_coef, None, None, False, cv2.SOLVEPNP_EPNP)
    # Using cv2.Rodrigues() to convert rvec from a rotation vector to a rotation matrix
    rvec_matrix, jacobian = cv2.Rodrigues(rvec)
    # Constructing the Mt matrix (camera extrinsics)
    Mt = np.concatenate((rvec_matrix, tvec), 1)
    np.set_printoptions(suppress=True)
    print("\n###### Mt Matrix ######")
    print(Mt)
    if plot:
        # Verifying Mt matrix by projecting the 3D points from world frame -> camera frame -> image plane
        reconstructed_image_points, jacobian = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, None)
        # Visualize the differences between original 2D points and reconstructed points
        image_points_new = image_points.reshape(-1, 2)
        x = image_points_new[:, 0]
        y = image_points_new[:, 1]
        reconstructed_image_points_new = reconstructed_image_points.reshape(-1, 2)
        x_recon = reconstructed_image_points_new[:, 0]
        y_recon = reconstructed_image_points_new[:, 1]
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.scatter(x, y, c='b', marker="s", label='Original Virtual Points')
        ax1.scatter(x_recon, y_recon, c='r', marker="o", label='Reconstructed Virtual Points')
        plt.legend(loc='upper right');
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Projection of Pi unto virtual plane')
        plt.savefig('Deltas.png')
        plt.show()
    return Mt

# Pi_list = np.loadtxt("20210413-103804_world_locations.txt")
# gi_list = np.loadtxt("20210413-103804_gaze_points.txt")
#
# # Defining camera_matrix as an identity matrix of size 3x3 - This is the camera intrinsics of the Pupil world camera
# # Matrix was obtained from Pupil Github repo
# # https://github.com/pupil-labs/pupil/blob/master/pupil_src/shared_modules/camera_models.py
# camera_matrix = np.array([[794.3311439869655, 0.0, 633.0104437728625],
#                 [0.0, 793.5290139393004, 397.36927353414865],
#                 [0.0, 0.0, 1.0]])
#
# # Defining zero distortion coefficients of Pupil world camera
# dist_coef = np.array([ -0.3758628065070806,
#                     0.1643326166951343,
#                     0.00012182540692089567,
#                     0.00013422608638039466,
#                     0.03343691733865076,
#                     0.08235235770849726,
#                     -0.08225804883227375,
#                     0.14463365333602152])
#
# object_points = np.array(Pi_list, dtype=np.float32)
# image_points = np.float32(gi_list)
# Mt = solve_pnp(object_points, image_points,camera_matrix, dist_coef)
#

