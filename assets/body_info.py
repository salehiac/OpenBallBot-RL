import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
import pdb

import utils

class OmniWheelRef:
    """
    Defines the referential constructed from the tree tangents between the omniwheels and the ball. It's interesting to see that this is actually a 2d referential (think about
    when the robot is in the init pose with its body - i.e. tower - aligning with z_global), and that it's only through rotations of the body that it can cover the 3d space

    It's mainly used for orientation, its origin is set at zero here but can be set depending on needs

    Noting (l_0, l_1, l2) this referential, we have

    (l_0 l_1 l_2) = (e_1^b e_2^b) M #naturally, M is 2x3

    where e_1^b and e_2^b are the two first vectors from the **body** (i.e. tower) referential e_1^b, e_2^b, e_3^b such that when the robot
    body is in the "init pose", we have e_3^b=e_3 (from global coords) and (e_1^b, e_2^b)=(e_1, e_2).

    Since the omniwheels are rigidly attached to the body, the matrix M will not change.

    ===========
    Change of coordinates: Note that M is not invertible, so we have to use least squares to map from vectors expressed relative to the e_i to their expression relative to the l_i. Let v be a vector 
    expressed relative to the (e_1^b, e_2^b) basis (e.g output of a pid), as

    (e_1^b e_2^b) v 

    and say we want to find u such that 

    (l_0 l_1 l_2) u = (e_1^b e_2^b) v

    From the equalities above this gives us 

    Mu=v

    and so we have to solve the least square problem argmin_u ||Mu-v||. However, since M is of shape 2*3, we can NOT use u=(M^TM)^{-1}M^Tv because M^TM is not invertible. 
    However, assuming M has full row rank (i.e. 2), we can use this expression for its pseudo-inverse:
                      M^{+}=M^T(MM^t)^{-1}
                      so u=M^{+}v
    of we can use tikhonov regularization instead with usual least squares: u=(M^TM+lambda*I)^{-1}M^tv

    Since the M that we have is well-conditioned (see self.cond), we'll just use the pseudo-inverse
    """

    def __init__(self):

        #lines segments taken from freecad (from edges of the motor mount that are parallel to the tangent between the wheels and the balls)
        self.seg_0=np.array([
            [1.035900873316046, -77.70464664175427, -67.79036799921596],#pt1
            [-33.13314618176402, -70.12333914583223, -67.79036762471033]])#pt2
    
        self.seg_1=np.array([
            [66.74653064118057, 39.8083348339609, -67.79036857993444],#pt1
            [77.30598420772867, 6.439218268251285, -67.79036861952144]])#pt2
    
        self.seg_2=np.array([
            [-67.77841493987036, 38.01009483758979, -67.7903686362132],#pt1 
            [-44.12671837519231, 63.80926632840315, -67.79036821659324]])#pt2

        #three vectors that will define a basis
        self.dir_0=(self.seg_0[1,:]-self.seg_0[0,:]).reshape(1,3)
        self.dir_1=(self.seg_1[1,:]-self.seg_1[0,:]).reshape(1,3)
        self.dir_2=(self.seg_2[1,:]-self.seg_2[0,:]).reshape(1,3)

        self.dir_0/=np.linalg.norm(self.dir_0)
        self.dir_1/=np.linalg.norm(self.dir_1)
        self.dir_2/=np.linalg.norm(self.dir_2)
         
        self.M=np.array([
            [self.dir_0[0,0], self.dir_0[0,1]],
            [self.dir_1[0,0], self.dir_1[0,1]],
            [self.dir_2[0,0], self.dir_2[0,1]]
            ]).transpose()

        MMt=self.M.dot(self.M.transpose())
        self.M_pseudoinv=self.M.transpose().dot(np.linalg.inv(MMt))

        if 0:
            utils.plot_vectors(origins=np.zeros([1,3]),
                     directions=np.concatenate([self.dir_0,self.dir_1,self.dir_2],0),
                     colors=["r","g","b"],
                     fig_ax=None,
                     axis_limits=[2,2,2],
                     scale_factor=1.0,
                     no_pause=True)
            plt.show()
            plt.close()
    
    def cond(self):
        """
        condition number of M
        """
        return np.linalg.cond(self.M)

    def body_plane_to_control_space(self, v):
        """
        v should be of shape 2*1
        """

        return self.M_pseudoinv.dot(v)

    def control_space_to_body_plane(self,u):
        """
        u should be of shape 3*1
        """

        return self.M.dot(u)


if __name__=="__main__":
    _omni=OmniWheelRef()

    _a=np.random.rand(2,1)
    _b=_omni.body_plane_to_control_space(_a)
    _c=_omni.control_space_to_body_plane(_b)

    print("condition matrix (M)==",_omni.cond())
    print("original==\n",_a)
    print("recovered==\n",_c)
    print("err_norm==",np.linalg.norm(_a-_c))




