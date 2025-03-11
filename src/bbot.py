import mujoco
import mujoco.viewer
import numpy as np
import sys
import pdb
import time
from termcolor import colored
import matplotlib.pyplot as plt
from typing import List, Optional


def plot_vectors(origins: np.ndarray,
                 directions: np.ndarray,
                 colors: List[str],
                 fig_ax: Optional[List] = None,
                 axis_limits: List[float] = None,
                 scale_factor: float = 1.0,
                 clear: bool = True,
                 dashed: bool = False):
    """
    origins  should be of shape num_pts*3, optionally, it can also be 1*3. In that case, all vectors will have the same origin
    directions  should be of shape num_pts*3
    """
    if fig_ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = fig_ax[0]
        ax = fig_ax[1]

    if clear:
        ax.clear()
    ax.quiver(*(origins.T * scale_factor),
              *(directions.T * scale_factor),
              color=colors,
              linestyle="solid" if not dashed else "dashed")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    axis_limits = axis_limits if axis_limits is not None else [10.0] * 3
    ax.set_xlim([-axis_limits[0], axis_limits[0]])
    ax.set_ylim([-axis_limits[1], axis_limits[1]])
    ax.set_zlim([-axis_limits[2], axis_limits[2]])

    plt.pause(0.01)
    return [fig, ax]


# Load the model
model = mujoco.MjModel.from_xml_path(sys.argv[1])
data = mujoco.MjData(model)
#model.opt.gravity = (0, 0, 0)

#print masses
total_mass = 0
for b_i in range(model.nbody):
    body_name = model.body(b_i).name
    cur_mass = model.body(body_name).mass.item()
    print(f"{body_name}: {cur_mass}")
    total_mass += cur_mass
print(colored(f"total_mass: {total_mass}", "magenta", attrs=["bold"]))

if 1:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.flags[
            mujoco.mjtVisFlag.
            mjVIS_JOINT] = False #freejoints are showed as boxes that mask the hinge ones, so comment the former in the xml file for debug
        viewer.opt.flags[mujoco.mjtVisFlag.
                         mjVIS_CONTACTPOINT] = False  # Show contact points
        viewer.opt.flags[mujoco.mjtVisFlag.
                         mjVIS_CONTACTFORCE] = False  # Show contact forces
        viewer.opt.flags[
            mujoco.mjtVisFlag.
            mjVIS_CONVEXHULL] = True  # Show collision bounding boxes

        fig_ax = None
        step_counter = 0
        plt.ion()
        while viewer.is_running():

            for ii in range(data.ncon):
                contact = data.contact[ii]
                contact_point = contact.pos
                tangent_1 = contact.frame[0:3]*contact.friction[0]
                tangent_2 = contact.frame[3:6]*contact.friction[1]
                normal = contact.frame[6:9]

                #first plot axes
                fig_ax = plot_vectors(origins=np.array([[0, 0, 0]]),
                                      directions=np.eye(3),
                                      colors=["r", "g", "b"],
                                      fig_ax=fig_ax,
                                      scale_factor=10,
                                      clear=True,
                                      dashed=True)

                #now plot the tangential contact forces
                origins = np.array(contact_point).reshape(
                    1, -1)  #all three vectors have the same origin
                directions = np.concatenate(
                    [tangent_1.reshape(1, -1),
                     tangent_2.reshape(1, -1)], 0)
                fig_ax = plot_vectors(origins=origins,
                                      directions=directions,
                                      colors=["k", "m"],
                                      fig_ax=fig_ax,
                                      scale_factor=10,
                                      clear=False)
                np.set_printoptions(precision=3,suppress=True)
                print(contact.friction)
                #print(step_counter)

            mujoco.mj_step(model, data)
            step_counter += 1
            viewer.sync()
            time.sleep(0.01)
            #pdb.set_trace()
