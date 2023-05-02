# import modules from orix (python-MTEX)
from diffpy.structure import Lattice, Structure
import matplotlib.pyplot as plt
import numpy as np
from orix.crystal_map import Phase
from orix.quaternion import Orientation, Rotation, symmetry
from orix.vector import AxAngle, Miller, Vector3d
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.pyplot import figure
from hexrd import rotations as rot
from hexrd import matrixutil as mutil
from hexrd import config
from hexrd import instrument


def Variant_Identifier(row):
    # Specify two crystal structures and symmetries
    phase1 = Phase(point_group="m-3m",structure=Structure(lattice=Lattice(3.58, 3.58, 3.58, 90, 90, 90)),)
    phase2 = Phase(point_group="6/mmm",structure=Structure(lattice=Lattice(2.54, 2.54, 4.1, 90, 90, 120)),)

    # Specify plane normal in the first crystal
    vec_1 = Miller(hkl=[[1, 1, 1],[-1, 1, 1],[1, 1, -1],[1, -1, 1]], phase=phase1)

    # Express the same directions with respect to the second crystal
    vec_2 = Miller(hkil=[[0, 0, 0, 2]], phase=phase2)

    if row['Transformed_epsilon'] == 1:
        # Create numpy arrays with exp_maps of austenite (fcc) and epsilon martensite (hcp)
        expMap_a = np.array([[row['exp_map_c[0]'], row['exp_map_c[1]'], row['exp_map_c[2]']]])
        expMap_e = np.array([[row['exp_map_c0_ep'], row['exp_map_c1_ep'], row['exp_map_c2_ep']]])

        # Turn exponential map into an orientation matrix (hcp)
        Ori_e=rot.rotMatOfExpMap(expMap_e.reshape(3,))
        o_e = Orientation.from_matrix(Ori_e,phase2.point_group)
        # fcc
        Ori_a=rot.rotMatOfExpMap(expMap_a.reshape(3,))
        o_a = Orientation.from_matrix(Ori_a,phase1.point_group)
        
        v_1_ori = Vector3d(~o_a * vec_1)
        v_2_ori = Vector3d(~o_e * vec_2)

        ang_1 = v_1_ori.unit
        ang_2 = v_2_ori.unit
        comp_1 = abs(ang_1.dot(ang_2)) #cos
        angle_radians = np.arccos(comp_1)
        angle_degrees = np.degrees(angle_radians)

        # Find the smallest angle
        min_angle = np.min(angle_degrees)

        # If the minimum value is less than the threshold, return the position of that minimum value in the array
        threshold = 10   # Can be changed default 5 degree to allow some grain rotation during deformation
        if min_angle < threshold:  
            min_index = np.argmin(angle_degrees) + 1
            print(min_angle)
            return min_index, min_angle
        else:
            return np.nan, np.nan
    else:
        return np.nan, np.nan

# Apply the Variant_Identifier function to each row of the DataFrame
new_df['Epsilon_variant'], new_df['Dev_ang_SN'] = zip(*new_df.apply(Variant_Identifier, axis=1))
