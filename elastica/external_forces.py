__doc__ = """ Numba implementation module for boundary condition implementations that apply
external forces to the system."""

from typing import TypeVar, Generic

import numpy as np
from numpy.typing import NDArray

from elastica._linalg import _batch_matvec
from elastica.typing import SystemType, RodType, RigidBodyType
from elastica.utils import _bspline

from numba import njit
from elastica._linalg import _batch_product_i_k_to_ik


S = TypeVar("S")


class NoForces(Generic[S]):
    """
    This is the base class for external forcing boundary conditions applied to rod-like objects.

    Notes
    -----
    Every new external forcing class must be derived
    from NoForces class.

    """

    def __init__(self) -> None:
        """
        NoForces class does not need any input parameters.
        """
        pass

    def apply_forces(self, system: S, time: np.float64 = np.float64(0.0)) -> None:
        """Apply forces to a rod-like object.

        In NoForces class, this routine simply passes.

        Parameters
        ----------
        system : SystemType
            Rod or rigid-body object
        time : float
            The time of simulation.

        """
        pass

    def apply_torques(self, system: S, time: np.float64 = np.float64(0.0)) -> None:
        """Apply torques to a rod-like object.

        In NoForces class, this routine simply passes.

        Parameters
        ----------
        system : SystemType
            Rod or rigid-body object
        time : float
            The time of simulation.

        """
        pass


class GravityForces(NoForces):
    """
    This class applies a constant gravitational force to the entire rod.

        Attributes
        ----------
        acc_gravity: numpy.ndarray
            1D (dim) array containing data with 'float' type. Gravitational acceleration vector.

    """

    def __init__(
        self,
        acc_gravity: NDArray[np.float64] = np.array(
            [0.0, -9.80665, 0.0]
        ),  # FIXME: avoid mutable default
    ) -> None:
        """

        Parameters
        ----------
        acc_gravity: numpy.ndarray
            1D (dim) array containing data with 'float' type. Gravitational acceleration vector.

        """
        super(GravityForces, self).__init__()
        self.acc_gravity = acc_gravity

    def apply_forces(
        self, system: "RodType | RigidBodyType", time: np.float64 = np.float64(0.0)
    ) -> None:
        self.compute_gravity_forces(
            self.acc_gravity, system.mass, system.external_forces
        )

    @staticmethod
    @njit(cache=True)  # type: ignore
    def compute_gravity_forces(
        acc_gravity: NDArray[np.float64],
        mass: NDArray[np.float64],
        external_forces: NDArray[np.float64],
    ) -> None:
        """
        This function add gravitational forces on the nodes. We are
        using njit decorated function to increase the speed.

        Parameters
        ----------
        acc_gravity: numpy.ndarray
            1D (dim) array containing data with 'float' type. Gravitational acceleration vector.
        mass: numpy.ndarray
            1D (blocksize) array containing data with 'float' type. Mass on the nodes.
        external_forces: numpy.ndarray
            2D (dim, blocksize) array containing data with 'float' type. External force vector.

        """
        inplace_addition(external_forces, _batch_product_i_k_to_ik(acc_gravity, mass))


class EndpointForces(NoForces):
    """
    This class applies constant forces on the endpoint nodes.

        Attributes
        ----------
        start_force: numpy.ndarray
            1D (dim) array containing data with 'float' type. Force applied to first node of the system.
        end_force: numpy.ndarray
            1D (dim) array containing data with 'float' type. Force applied to last node of the system.
        ramp_up_time: float
            Applied forces are ramped up until ramp up time.

    """

    def __init__(
        self,
        start_force: NDArray[np.float64],
        end_force: NDArray[np.float64],
        ramp_up_time: float,
    ) -> None:
        """

        Parameters
        ----------
        start_force: numpy.ndarray
            1D (dim) array containing data with 'float' type.
            Force applied to first node of the system.
        end_force: numpy.ndarray
            1D (dim) array containing data with 'float' type.
            Force applied to last node of the system.
        ramp_up_time: float
            Applied forces are ramped up until ramp up time.

        """
        super(EndpointForces, self).__init__()
        self.start_force = start_force
        self.end_force = end_force
        assert ramp_up_time > 0.0
        self.ramp_up_time = np.float64(ramp_up_time)

    def apply_forces(
        self, system: "RodType | RigidBodyType", time: np.float64 = np.float64(0.0)
    ) -> None:
        self.compute_end_point_forces(
            system.external_forces,
            self.start_force,
            self.end_force,
            time,
            self.ramp_up_time,
        )

    @staticmethod
    @njit(cache=True)  # type: ignore
    def compute_end_point_forces(
        external_forces: NDArray[np.float64],
        start_force: NDArray[np.float64],
        end_force: NDArray[np.float64],
        time: np.float64,
        ramp_up_time: np.float64,
    ) -> None:
        """
        Compute end point forces that are applied on the rod using numba njit decorator.

        Parameters
        ----------
        external_forces: numpy.ndarray
            2D (dim, blocksize) array containing data with 'float' type. External force vector.
        start_force: numpy.ndarray
            1D (dim) array containing data with 'float' type.
        end_force: numpy.ndarray
            1D (dim) array containing data with 'float' type.
            Force applied to last node of the system.
        time: float
        ramp_up_time: float
            Applied forces are ramped up until ramp up time.

        """
        factor = min(1.0, float(time / ramp_up_time))
        external_forces[..., 0] += start_force * factor
        external_forces[..., -1] += end_force * factor


class UniformTorques(NoForces):
    """
    This class applies a uniform torque to the entire rod.

        Attributes
        ----------
        torque: numpy.ndarray
            2D (dim, 1) array containing data with 'float' type. Total torque applied to a rod-like object.

    """

    def __init__(
        self,
        torque: np.float64,
        direction: NDArray[np.float64] = np.array(
            [0.0, 0.0, 0.0]
        ),  # FIXME: avoid mutable default
    ) -> None:
        """

        Parameters
        ----------
        torque: float
            Torque magnitude applied to a rod-like object.
        direction: numpy.ndarray
            1D (dim) array containing data with 'float' type.
            Direction in which torque applied.
        """
        super(UniformTorques, self).__init__()
        self.torque = torque * direction

    def apply_torques(
        self, system: "RodType | RigidBodyType", time: np.float64 = np.float64(0.0)
    ) -> None:
        n_elems = system.n_elems
        torque_on_one_element = (
            _batch_product_i_k_to_ik(self.torque, np.ones((n_elems))) / n_elems
        )
        system.external_torques += _batch_matvec(
            system.director_collection, torque_on_one_element
        )


class UniformForces(NoForces):
    """
    This class applies a uniform force to the entire rod.

        Attributes
        ----------
        force:  numpy.ndarray
            2D (dim, 1) array containing data with 'float' type. Total force applied to a rod-like object.
    """

    def __init__(
        self,
        force: np.float64,
        direction: NDArray[np.float64] = np.array(
            [0.0, 0.0, 0.0]
        ),  # FIXME: avoid mutable default
    ) -> None:
        """

        Parameters
        ----------
        force: float
            Force magnitude applied to a rod-like object.
        direction: numpy.ndarray
            1D (dim) array containing data with 'float' type.
            Direction in which force applied.
        """
        super(UniformForces, self).__init__()
        self.force = (force * direction).reshape(3, 1)

    def apply_forces(
        self, system: "RodType | RigidBodyType", time: np.float64 = np.float64(0.0)
    ) -> None:
        force_on_one_element = self.force / system.n_elems

        system.external_forces += force_on_one_element

        # Because mass of first and last node is half
        system.external_forces[..., 0] -= 0.5 * force_on_one_element[:, 0]
        system.external_forces[..., -1] -= 0.5 * force_on_one_element[:, 0]


class MuscleTorques(NoForces):
    """
    This class applies muscle torques along the body. The applied muscle torques are treated
    as applied external forces. This class can apply
    muscle torques as a traveling wave with a beta spline or only
    as a traveling wave. For implementation details refer to Gazzola et. al.
    RSoS. (2018).

        Attributes
        ----------
        direction: numpy.ndarray
            2D (dim, 1) array containing data with 'float' type. Muscle torque direction.
        angular_frequency: float
            Angular frequency of traveling wave.
        wave_number: float
            Wave number of traveling wave.
        phase_shift: float
            Phase shift of traveling wave.
        ramp_up_time: float
            Applied muscle torques are ramped up until ramp up time.
        my_spline: numpy.ndarray
            1D (blocksize) array containing data with 'float' type. Generated spline.

    """

    def __init__(
        self,
        base_length: float,  # TODO: Is this necessary?
        b_coeff: NDArray[np.float64],
        period: float,
        wave_number: float,
        phase_shift: float,
        direction: NDArray[np.float64],
        rest_lengths: NDArray[np.float64],
        ramp_up_time: float,
        with_spline: bool = False,
    ) -> None:
        """

        Parameters
        ----------
        base_length: float
            Rest length of the rod-like object.
        b_coeff: nump.ndarray
            1D array containing data with 'float' type.
            Beta coefficients for beta-spline.
        period: float
            Period of traveling wave.
        wave_number: float
            Wave number of traveling wave.
        phase_shift: float
            Phase shift of traveling wave.
        direction: numpy.ndarray
           1D (dim) array containing data with 'float' type. Muscle torque direction.
        ramp_up_time: np.float64
            Applied muscle torques are ramped up until ramp up time.
        with_spline: boolean
            Option to use beta-spline.

        """
        super(MuscleTorques, self).__init__()

        self.direction = direction  # Direction torque applied
        self.angular_frequency = np.float64(2.0 * np.pi / period)
        self.wave_number = np.float64(wave_number)
        self.phase_shift = np.float64(phase_shift)

        assert ramp_up_time > 0.0
        self.ramp_up_time = np.float64(ramp_up_time)

        # s is the position of nodes on the rod, we go from node=1 to node=nelem-1, because there is no
        # torques applied by first and last node on elements. Reason is that we cannot apply torque in an
        # infinitesimal segment at the beginning and end of rod, because there is no additional element
        # (at element=-1 or element=n_elem+1) to provide internal torques to cancel out an external
        # torque. This coupled with the requirement that the sum of all muscle torques has
        # to be zero results in this condition.
        self.s = np.cumsum(rest_lengths)
        self.s /= self.s[-1]

        if with_spline:
            assert b_coeff.size != 0, "Beta spline coefficient array (t_coeff) is empty"
            my_spline, ctr_pts, ctr_coeffs = _bspline(b_coeff)
            self.my_spline = my_spline(self.s)

        else:
            self.my_spline = np.full_like(self.s, fill_value=1.0)

    def apply_torques(
        self, system: "RodType | RigidBodyType", time: np.float64 = np.float64(0.0)
    ) -> None:
        self.compute_muscle_torques(
            time,
            self.my_spline,
            self.s,
            self.angular_frequency,
            self.wave_number,
            self.phase_shift,
            self.ramp_up_time,
            self.direction,
            system.director_collection,
            system.external_torques,
        )

    @staticmethod
    @njit(cache=True)  # type: ignore
    def compute_muscle_torques(
        time: float,
        my_spline: NDArray[np.float64],
        s: np.float64,
        angular_frequency: np.float64,
        wave_number: np.float64,
        phase_shift: np.float64,
        ramp_up_time: np.float64,
        direction: NDArray[np.float64],
        director_collection: NDArray[np.float64],
        external_torques: NDArray[np.float64],
    ) -> None:
        # Ramp up the muscle torque
        factor = min(1.0, float(time / ramp_up_time))
        # From the node 1 to node nelem-1
        # Magnitude of the torque. Am = beta(s) * sin(2pi*t/T + 2pi*s/lambda + phi)
        # There is an inconsistency with paper and Elastica cpp implementation. In paper sign in
        # front of wave number is positive, in Elastica cpp it is negative.
        torque_mag = (
            factor
            * my_spline
            * np.sin(angular_frequency * time - wave_number * s + phase_shift)
        )
        # Head and tail of the snake is opposite compared to elastica cpp. We need to iterate torque_mag
        # from last to first element.
        torque = _batch_product_i_k_to_ik(direction, torque_mag[::-1])
        inplace_addition(
            external_torques[..., 1:],
            _batch_matvec(director_collection, torque)[..., 1:],
        )
        inplace_substraction(
            external_torques[..., :-1],
            _batch_matvec(director_collection[..., :-1], torque[..., 1:]),
        )


@njit(cache=True)  # type: ignore
def inplace_addition(
    external_force_or_torque: NDArray[np.float64],
    force_or_torque: NDArray[np.float64],
) -> None:
    """
    This function does inplace addition. First argument
    `external_force_or_torque` is the system.external_forces
    or system.external_torques. Second argument force or torque
    vector to be added.

    Parameters
    ----------
    external_force_or_torque: numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type.
    force_or_torque: numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type.


    """
    blocksize = force_or_torque.shape[1]
    for i in range(3):
        for k in range(blocksize):
            external_force_or_torque[i, k] += force_or_torque[i, k]


@njit(cache=True)  # type: ignore
def inplace_substraction(
    external_force_or_torque: NDArray[np.float64],
    force_or_torque: NDArray[np.float64],
) -> None:
    """
    This function does inplace substraction. First argument
    `external_force_or_torque` is the system.external_forces
    or system.external_torques. Second argument force or torque
    vector to be substracted.
    Parameters
    ----------
    external_force_or_torque: numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type.
    force_or_torque: numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type.


    """
    blocksize = force_or_torque.shape[1]
    for i in range(3):
        for k in range(blocksize):
            external_force_or_torque[i, k] -= force_or_torque[i, k]


class EndpointForcesSinusoidal(NoForces):
    """
    This class applies sinusoidally varying forces to the ends of a rod.
    Forces are applied in a plane, which is defined by the tangent_direction and normal_direction.

        Attributes
        ----------
        start_force_mag: float
            Magnitude of the force that is applied to the start of the rod (node 0).
        end_force_mag: float
            Magnitude of the force that is applied to the end of the rod (node -1).
        ramp_up_time: float
            Applied forces are applied in the normal direction until time reaches ramp_up_time.
        normal_direction: np.ndarray
            An array (3,) contains type float.
            This is the normal direction of the rod.
        roll_direction: np.ndarray
            An array (3,) contains type float.
            This is the direction perpendicular to rod tangent, and rod normal.

        Notes
        -----
        In order to see example how to use this class, see joint examples.

    """

    def __init__(
        self,
        start_force_mag: float,
        end_force_mag: float,
        ramp_up_time: float = 0.0,
        tangent_direction: NDArray[np.floating] = np.array(
            [0.0, 0.0, 1.0]
        ),  # FIXME: avoid mutable default
        normal_direction: NDArray[np.floating] = np.array(
            [0.0, 1.0, 0.0]
        ),  # FIXME: avoid mutable default
    ) -> None:
        """

        Parameters
        ----------
        start_force_mag: float
            Magnitude of the force that is applied to the start of the system (node 0).
        end_force_mag: float
            Magnitude of the force that is applied to the end of the system (node -1).
        ramp_up_time: float
            Applied muscle torques are ramped up until ramp up time.
        tangent_direction: np.ndarray
            An array (3,) contains type float.
            This is the tangent direction of the system, or normal of the plane that forces applied.
        normal_direction: np.ndarray
            An array (3,) contains type float.
            This is the normal direction of the system.
        """
        super(EndpointForcesSinusoidal, self).__init__()
        # Start force
        self.start_force_mag = np.float64(start_force_mag)
        self.end_force_mag = np.float64(end_force_mag)

        # Applied force directions
        self.normal_direction = normal_direction
        self.roll_direction = np.cross(normal_direction, tangent_direction)

        assert ramp_up_time >= 0.0
        self.ramp_up_time = np.float64(ramp_up_time)

    def apply_forces(
        self, system: "RodType | RigidBodyType", time: np.float64 = np.float64(0.0)
    ) -> None:

        if time < self.ramp_up_time:
            # When time smaller than ramp up time apply the force in normal direction
            # First pull the rod upward or downward direction some time.
            start_force = -2.0 * self.start_force_mag * self.normal_direction
            end_force = -2.0 * self.end_force_mag * self.normal_direction

            system.external_forces[..., 0] += start_force
            system.external_forces[..., -1] += end_force

        else:
            # When time is greater than ramp up time, forces are applied in normal
            # and roll direction or forces are in a plane perpendicular to the
            # direction.

            # First force applied to start of the rod
            roll_forces_start = (
                self.start_force_mag
                * np.cos(0.5 * np.pi * (time - self.ramp_up_time))
                * self.roll_direction
            )
            normal_forces_start = (
                self.start_force_mag
                * np.sin(0.5 * np.pi * (time - self.ramp_up_time))
                * self.normal_direction
            )
            start_force = roll_forces_start + normal_forces_start
            # Now force applied to end of the rod
            roll_forces_end = (
                self.end_force_mag
                * np.cos(0.5 * np.pi * (time - self.ramp_up_time))
                * self.roll_direction
            )
            normal_forces_end = (
                self.end_force_mag
                * np.sin(0.5 * np.pi * (time - self.ramp_up_time))
                * self.normal_direction
            )
            end_force = roll_forces_end + normal_forces_end
            # Update external forces
            system.external_forces[..., 0] += start_force
            system.external_forces[..., -1] += end_force

class TendonForces(NoForces):
    """
    This class applies tendon forcing along the length of the rod.

        Attributes
        ----------
        vertebra_height: float
            Height at which the tendon contacts the vertebra. It should be the highest point on the tendon-vertebra space.
        num_vertebrae: int
            Amount of vertebrae to be used in the system.
        vertebra_height_vector: numpy.ndarray
            1D (dim) numpy array. Describes the orientation and height in space of the vertebrae in the system.
        tension: float
            Tension applied to the tendon in the system.
        n_elements: int
            Total amount of nodes in the rod system. This value is set in the simulator and is copied to this class for later use.
        vertebra_nodes: list
            1D (dim) list. Contains the node numbers of every node with vertebrae. The vertebrae are assumed to be uniformly spaced through the intervals specified by 
            first_vertebra_node and final_vertebra_node, with an amount equal to num_vertebrae.
        force_data: numpy.ndarray
            2D (dim,3) numpy array. Contains the force vectors caused by tendon forcing for each of the nodes with vertebrae.
        

    """

    def __init__(self, vertebra_height, num_vertebrae, first_vertebra_node, final_vertebra_node, tension, vertebra_height_orientation, n_elements):
        """

        Parameters 
        ----------
        vertebra_height: float
            Height at which the tendon contacts the vertebra. It should be the highest point on the tendon-vertebra space.
        num_vertebrae: int
            Amount of vertebrae to be used in the system (aka num of disks).
        first_vertebra_node: int
            The first node to have a vertebra, from the base of the rod to the tip.
        final_vertebra_node: int
            The last node to have a vertebra, from the base of the rod to the tip.
        vertebra_mass: float
            Total mass of a single vertebra.
        tension: float
            Tension applied to the tendon in the system.
        vertebra_height_orientation: numpy.ndarray
            1D (dim) numpy array. Describes the orientatation of the vertebrae in the system.
        n_elements: int
            Total amount of nodes in the rod system. This value is set in the simulator and is copied to this class for later use.
        """
        super(TendonForces, self).__init__()

        # Initializing class attributes
        self.vertebra_height = vertebra_height
        self.num_vertebrae = num_vertebrae
        self.vertebra_height_vector_local = vertebra_height_orientation * vertebra_height # in local reference frame
        self.tension = tension
        self.n_elements = n_elements

        # Creating vertebra node indices (node-indexed)
        self.vertebra_nodes = []
        vertebra_increment = (final_vertebra_node - first_vertebra_node) / (num_vertebrae - 1)
        for i in range(num_vertebrae):
            idx = int(round(i * vertebra_increment + first_vertebra_node))
            # clamp the values to valid node range
            idx = max(0, min(idx, self.n_elements))
            self.vertebra_nodes.append(idx)

    def apply_forces(self, system: SystemType, time: np.float64 = 0.0):
         # The application of the force data is done outside of the @njit decorated function because self.force_data needs to be referenced in self.compute_torques()

        # Retrieves relative position unit norm vectors between each vertebra top (where the tendon contacts the vertebra)
        unit_norm_vector_array = self.get_rotations(np.array(system.position_collection), np.array(system.director_collection), np.array(self.vertebra_nodes), np.array(self.vertebra_height_vector_local))

        # Computes forces in global frame (forces in each vertebra)
        self.force_data = self.compute_forces(self.tension, np.array(self.vertebra_nodes), unit_norm_vector_array)

        # Creating force data set to apply to the rod () forces into the simulator's global external_forces array
        apply_force = np.zeros((3, self.n_elements + 1), dtype=np.float64)

        for i, node_idx in enumerate(self.vertebra_nodes):
            apply_force[:, node_idx] = self.force_data[i]

        # Applies forces to the rod system
        system.external_forces += apply_force

    def apply_torques(self, system: SystemType, time: np.float64 = 0.0):
        # Force_data set is expressed in the global coordinate frame and must be changed to the local reference frame to calculate torque
        transformed_force_data = np.zeros((len(self.vertebra_nodes), 3), dtype=np.float64)

        # Transforming the force vectors calculated in compute_forces method from global reference frame to local reference frame
        for i, node_idx in enumerate(self.vertebra_nodes):
            elem_idx = min(node_idx, self.n_elements - 1)
            R = system.director_collection[..., elem_idx]  # element rotation matrix
            transformed_force_data[i] = R.T @ self.force_data[i]

        self.compute_torques(
            self.vertebra_height_vector_local, self.vertebra_nodes, transformed_force_data,
            self.n_elements, system.external_torques
        )

    @staticmethod
    @njit(cache=True)
    def get_rotations(position_collection, director_collection, vertebra_nodes, vertebra_height_vector_local):
        # Returns an array containing the unit norm vector which describes the orientation of each segment of tendon between vertebrae

        # Initializing unit_norm_vector_array to store the unit normed vectors that describe the global orientation of the forces in each vertebra
        num_nodes = len(vertebra_nodes)
        unit_norm_vector_array = np.zeros((num_nodes + 1, 3), dtype=np.float64)

        for i in range(num_nodes + 1):
            # There is a +1 in the for loop to account for the force between the first vertebra and the fixed node

            # If statement, used for the case when i = 0 and thus there is no vertebra before this one, same for the final vertebra (no vertebra after that one)
            if i == 0:
                current_vertebra = 0
                next_vertebra = vertebra_nodes[0]
            elif i == num_nodes:
                current_vertebra = vertebra_nodes[num_nodes - 1]
                next_vertebra = vertebra_nodes[num_nodes - 1]
            else:
                current_vertebra = vertebra_nodes[i - 1]
                next_vertebra = vertebra_nodes[i]

            # Global node positions
            x_current = position_collection[0, current_vertebra]
            y_current = position_collection[1, current_vertebra]
            z_current = position_collection[2, current_vertebra]

            x_next = position_collection[0, next_vertebra]
            y_next = position_collection[1, next_vertebra]
            z_next = position_collection[2, next_vertebra]

            current_rotation_matrix = director_collection[..., current_vertebra if current_vertebra <= director_collection.shape[2]-1 else director_collection.shape[2]-1]
            next_rotation_matrix = director_collection[..., next_vertebra if next_vertebra <= director_collection.shape[2]-1 else director_collection.shape[2]-1]

            current_node = np.array([x_current, y_current, z_current], dtype=np.float64)
            next_node = np.array([x_next, y_next, z_next], dtype=np.float64)

            # Transforms local height into global (global_offset = R @ vertebra_height_vector_local)
            r_curr = np.ascontiguousarray(current_rotation_matrix) @ np.ascontiguousarray(vertebra_height_vector_local)
            r_next = np.ascontiguousarray(next_rotation_matrix) @ np.ascontiguousarray(vertebra_height_vector_local)

            delta_vector = (next_node + r_next) - (current_node + r_curr)

            # Calculating the unit-normed vector based on the differences calculated in the previous step
            delta_vector_norm = np.linalg.norm(delta_vector)
            if delta_vector_norm > 0.0:
                unit_norm_delta_vector = delta_vector / delta_vector_norm
            else:
                unit_norm_delta_vector = np.zeros(3, dtype=np.float64)

            # This if statement is to stop unit_norm_delta_vector from becoming a 'nan'
            if i == num_nodes:
                unit_norm_delta_vector = np.zeros(3, dtype=np.float64)
            
            # Storing the unit normed vector to be later used in the compute_forces method
            unit_norm_vector_array[i, :] = unit_norm_delta_vector

        return unit_norm_vector_array

    @staticmethod
    @njit(cache=True)
    def compute_forces(tension, vertebra_nodes, unit_norm_vector_array):
        # Creating array to store forces in vertebrae
        force_data = np.zeros((len(vertebra_nodes), 3), dtype=np.float64)

        for i in range(len(vertebra_nodes)):
            # This for loop multiplies the unit normed vectors calculated previously, with the tension of the tendon, thus creating the force vector for each vertebra
            # Contiguous array to increase speed in njit decorator
            force_current_prev = unit_norm_vector_array[i] * -tension
            force_current_next = unit_norm_vector_array[i + 1] * tension

            # Summing the components of both force vectors to get the final force vector, which is then stored for use in the apply_forces and compute_torques methods
            force_data[i, :] = force_current_prev + force_current_next

        return force_data

    @staticmethod
    @njit(cache=True)
    def compute_torques(vertebra_height_vector_local, vertebra_nodes, transformed_force_data, n_elements, external_torques):
        
        # Creating torque data set for storage
        torque_data = np.zeros((len(vertebra_nodes), 3), dtype=np.float64)

        # Goes through vertebra nodes to calculate torques for them
        for i in range(len(vertebra_nodes)):

            # Cross product between the vertebra height vector and the local force vector due to the tendons, to obtain the tendon torque for that vertebra
            torque_vector = np.cross(vertebra_height_vector_local, transformed_force_data[i])

            # Sum of the vectors, and storage into the torque_data array
            torque_data[i, :] = torque_vector

        # Appending the computed torque vector to the final torque data set
        apply_torque = np.zeros((3, n_elements + 1), dtype=np.float64)

        k = 0
        for node_idx in vertebra_nodes:
            elem_idx = node_idx
            if elem_idx >= n_elements:
                elem_idx = n_elements - 1
            apply_torque[:, elem_idx] = torque_data[k, :]
            k += 1

        # Applying the torque data set to the rod (torque on the final vertebra)
        external_torques += apply_torque

# class TendonForces(NoForces):
#     """
#     This class applies tendon forcing along the length of the rod.

#         Attributes
#         ----------
#         vertebra_height: float
#             Height at which the tendon contacts the vertebra. It should be the highest point on the tendon-vertebra space.
#         num_vertebrae: int
#             Amount of vertebrae to be used in the system.
#         vertebra_height_vector: numpy.ndarray
#             1D (dim) numpy array. Describes the orientation and height in space of the vertebrae in the system.
#         tension: float
#             Tension applied to the tendon in the system.
#         n_elements: int
#             Total amount of nodes in the rod system. This value is set in the simulator and is copied to this class for later use.
#         vertebra_nodes: list
#             1D (dim) list. Contains the node numbers of every node with vertebrae. The vertebrae are assumed to be uniformly spaced through the intervals specified by 
#             first_vertebra_node and final_vertebra_node, with an amount equal to num_vertebrae.
#         force_data: numpy.ndarray
#             2D (dim,3) numpy array. Contains the force vectors caused by tendon forcing for each of the nodes with vertebrae.
        

#     """

#     def __init__(self, vertebra_height, num_vertebrae, first_vertebra_node, final_vertebra_node, tension, vertebra_height_orientation, n_elements):
#         """

#         Parameters 
#         ----------
#         vertebra_height: float
#             Height at which the tendon contacts the vertebra. It should be the highest point on the tendon-vertebra space.
#         num_vertebrae: int
#             Amount of vertebrae to be used in the system.
#         first_vertebra_node: int
#             The first node to have a vertebra, from the base of the rod to the tip.
#         final_vertebra_node: int
#             The last node to have a vertebra, from the base of the rod to the tip.
#         vertebra_mass: float
#             Total mass of a single vertebra.
#         tension: float
#             Tension applied to the tendon in the system.
#         vertebra_height_orientation: numpy.ndarray
#             1D (dim) numpy array. Describes the orientatation of the vertebrae in the system.
#         n_elements: int
#             Total amount of nodes in the rod system. This value is set in the simulator and is copied to this class for later use.
#         """
#         super(TendonForces, self).__init__()

#         # Initializing class attributes to be used in other methods
#         self.vertebra_height = vertebra_height
#         self.num_vertebrae = num_vertebrae
#         self.vertebra_height_vector = vertebra_height_orientation * vertebra_height
#         self.tension = tension
#         self.n_elements = n_elements

#         # Creating vector containing the node numbers with the vertebras for this instance of TendonForces
#         self.vertebra_nodes = []
#         vertebra_increment = (final_vertebra_node - first_vertebra_node)/(num_vertebrae - 1)
#         for i in range(num_vertebrae):
#             self.vertebra_nodes.append(round(i * vertebra_increment + first_vertebra_node))

#     def apply_forces(self, system: SystemType, time: np.float64 = 0.0):
#         # The application of the force data is done outside of the @njit decorated function because self.force_data needs to be referenced in self.compute_torques()

#         # Retrieves relative position unit norm vectors between each vertebra top (where the tendon contacts the vertebra)
#         unit_norm_vector_array = self.get_rotations(np.array(system.position_collection), np.array(system.director_collection), np.array(self.vertebra_nodes), self.vertebra_height_vector)

#         # Computes the forces in each vertebra
#         self.force_data = self.compute_forces(self.tension, np.array(self.vertebra_nodes), unit_norm_vector_array)

#         # Creating the force data set to apply to the rod
#         apply_force = np.zeros((3,self.n_elements+1))

#         # PyElastica handles forces in GLOBAL coord. system, so they are applied directly. Also, the vertebra weights are added to each vertebra
#         for i in range (len(self.vertebra_nodes)):
#             apply_force[:,self.vertebra_nodes[i]] = self.force_data[i]

#         # Applies forces to the rod
#         system.external_forces += apply_force


#     def apply_torques(self, system: SystemType, time: np.float64 = 0.0):
#         # The force_data set and vertebra_weight_vector are expressed in the global coordinate frame and must be changed to local reference frames for torque application
#         # Creating the array which will contain the transformed force vectors
#         transformed_force_data = np.zeros((len(self.vertebra_nodes), 3), dtype=np.float64)

#         # Transforming the force vectors calculated in the compute_forces method from the global reference frame to the local reference frame
#         for i in range(len(self.vertebra_nodes)):
#             transformed_force_data[i] = system.director_collection[...,(self.vertebra_nodes[i]-1)] @ self.force_data[i]

#         self.compute_torques(
#             self.vertebra_height_vector, np.array(self.vertebra_nodes), transformed_force_data,
#             self.n_elements, system.external_torques
#         )


#     @staticmethod
#     @njit(cache=True)
#     def get_rotations(position_collection, director_collection, vertebra_nodes, vertebra_height_vector):
#         # Returns an array containing the unit norm vector which describes the orientation of each segment of tendon between vertebrae

#         # Initializing unit_norm_vector_array to store the unit normed vectors that describe the global orientation of the forces in each vertebra
#         unit_norm_vector_array = np.zeros((len(vertebra_nodes) + 1, 3), dtype=np.float64)

#         for i in range(len(vertebra_nodes)+1):
#             # There is a +1 in the for loop to account for the force between the first vertebra and the fixed node

#             # If statement, used for the case when i = 0 and thus there is no vertebra before this one, same for the final vertebra (no vertebra after that one)
#             if i==0:
#                 current_vertebra = 0
#                 next_vertebra = vertebra_nodes[i]
#             elif i==len(vertebra_nodes):
#                 current_vertebra = vertebra_nodes[i-1]
#                 next_vertebra = vertebra_nodes[i-1]
#             else:
#                 current_vertebra = vertebra_nodes[i-1]
#                 next_vertebra = vertebra_nodes[i]

#             # Setting up values to be used iteratively
#             x_current = position_collection[0, current_vertebra]
#             y_current = position_collection[1, current_vertebra]
#             z_current = position_collection[2, current_vertebra]

#             x_next = position_collection[0, next_vertebra]
#             y_next = position_collection[1, next_vertebra]
#             z_next = position_collection[2, next_vertebra]

#             current_rotation_matrix = director_collection[...,current_vertebra]
#             next_rotation_matrix = director_collection[...,next_vertebra]

#             current_node = np.array([x_current, y_current, z_current])
#             next_node = np.array([x_next, y_next, z_next])

#             # Calculating relative position vector between vertebrae, considering the vertebra height
#             # Continguous arrays to help with computation speed
#             delta_vector = (next_node + np.ascontiguousarray(next_rotation_matrix.T) @ np.ascontiguousarray(vertebra_height_vector)) - (current_node + np.ascontiguousarray(current_rotation_matrix.T) @ np.ascontiguousarray(vertebra_height_vector))

#             # Calculating the unit-normed vector based on the differences calculated in the previous step
#             delta_vector_norm = np.linalg.norm(delta_vector)
#             unit_norm_delta_vector = delta_vector / delta_vector_norm

#             # This if statement is to stop unit_norm_delta_vector from becoming a 'nan'
#             if i==len(vertebra_nodes):
#                 unit_norm_delta_vector = np.zeros(3)

#             # Storing the unit normed vector to be later used in the compute_forces method
#             unit_norm_vector_array[i] = unit_norm_delta_vector

#         return unit_norm_vector_array

#     @staticmethod
#     @njit(cache=True)
#     def compute_forces(tension, vertebra_nodes, unit_norm_vector_array):

#         # Creating array to store forces in vertebrae
#         force_data = np.zeros((len(vertebra_nodes), 3), dtype=np.float64)

#         for i in range(len(vertebra_nodes)):
#             # This for loop multiplies the unit normed vectors calculated previously, with the tension of the tendon, thus creating the force vector for each vertebra
#             # Contiguous array to increase speed in njit decorator
#             force_current_prev = unit_norm_vector_array[i] * -tension
#             force_current_next = unit_norm_vector_array[i+1] * tension

#             # Summing the components of both force vectors to get the final force vector, which is then stored for use in the apply_forces and compute_torques methods
#             force_data[i] = force_current_prev + force_current_next

#         return force_data


#     @staticmethod
#     @njit(cache=True)
#     def compute_torques(vertebra_height_vector, vertebra_nodes, transformed_force_data, n_elements, external_torques):

#         # Creating torque data set for storage
#         torque_data = np.zeros((len(vertebra_nodes), 3),dtype=np.float64)

#         # Goes through vertebra nodes to calculate torques for them
#         for i in range(len(vertebra_nodes)):

#             # Cross product between the vertebra height vector and the local force vector due to the tendons, to obtain the tendon torque for that vertebra
#             torque_vector = np.cross(vertebra_height_vector, transformed_force_data[i])

#             # Sum of the vectors, and storage into the torque_data array
#             torque_data[i] = torque_vector

#         # Appending the computed torque vector to the final torque data set
#         apply_torque = np.zeros((3,n_elements+1))

#         k = 0
#         for i in range(n_elements):
#             if i in vertebra_nodes:
#                 apply_torque[:,i] = torque_data[k]
#                 k += 1
#         apply_torque = apply_torque[:,1:]

#         # Applying the torque data set to the rod (torque on the final vertebra)
#         external_torques += apply_torque


class TendonForcesGravity(NoForces):
    """
    This class applies tendon forcing along the length of the rod and considers the gravity of the disks.

        Attributes
        ----------
        vertebra_height: float
            Height at which the tendon contacts the vertebra. It should be the highest point on the tendon-vertebra space.
        num_vertebrae: int
            Amount of vertebrae to be used in the system.
        vertebra_height_vector: numpy.ndarray
            1D (dim) numpy array. Describes the orientation and height in space of the vertebrae in the system.
        tension: float
            Tension applied to the tendon in the system.
        n_elements: int
            Total amount of nodes in the rod system. This value is set in the simulator and is copied to this class for later use.
        vertebra_weight_vector: numpy.ndarray
            1D (dim) numpy array. Vector which specifies the orientation and magnitude of the weight of the vertebrae (By default it is in the global -Z direction).
        vertebra_nodes: list
            1D (dim) list. Contains the node numbers of every node with vertebrae. The vertebrae are assumed to be uniformly spaced through the intervals specified by 
            first_vertebra_node and final_vertebra_node, with an amount equal to num_vertebrae.
        force_data: numpy.ndarray
            2D (dim,3) numpy array. Contains the force vectors caused by tendon forcing for each of the nodes with vertebrae.
        

    """

    def __init__(self, vertebra_height, num_vertebrae, first_vertebra_node, final_vertebra_node, vertebra_mass, tension, vertebra_height_orientation, n_elements):
        """

        Parameters 
        ----------
        vertebra_height: float
            Height at which the tendon contacts the vertebra. It should be the highest point on the tendon-vertebra space.
        num_vertebrae: int
            Amount of vertebrae to be used in the system.
        first_vertebra_node: int
            The first node to have a vertebra, from the base of the rod to the tip.
        final_vertebra_node: int
            The last node to have a vertebra, from the base of the rod to the tip.
        vertebra_mass: float
            Total mass of a single vertebra.
        tension: float
            Tension applied to the tendon in the system.
        vertebra_height_orientation: numpy.ndarray
            1D (dim) numpy array. Describes the orientatation of the vertebrae in the system.
        n_elements: int
            Total amount of nodes in the rod system. This value is set in the simulator and is copied to this class for later use.
        """
        super(TendonForcesGravity, self).__init__()

        # Initializing class attributes to be used in other methods
        self.vertebra_height = vertebra_height
        self.num_vertebrae = num_vertebrae
        self.vertebra_height_vector = vertebra_height_orientation * vertebra_height
        self.tension = tension
        self.n_elements = n_elements
        self.vertebra_weight_vector = np.array([0.0, 0.0, -vertebra_mass * 9.80665])

        # Creating vector containing the node numbers with the vertebras for this instance of TendonForces
        self.vertebra_nodes = []
        vertebra_increment = (final_vertebra_node - first_vertebra_node)/(num_vertebrae - 1)
        for i in range(num_vertebrae):
            self.vertebra_nodes.append(round(i * vertebra_increment + first_vertebra_node))

    def apply_forces(self, system: SystemType, time: np.float64 = 0.0):
        # The application of the force data is done outside of the @njit decorated function because self.force_data needs to be referenced in self.compute_torques()

        # Retrieves relative position unit norm vectors between each vertebra top (where the tendon contacts the vertebra)
        unit_norm_vector_array = self.get_rotations(np.array(system.position_collection), np.array(system.director_collection), np.array(self.vertebra_nodes), self.vertebra_height_vector)

        # Computes the forces in each vertebra
        self.force_data = self.compute_forces(self.tension, np.array(self.vertebra_nodes), unit_norm_vector_array)

        # Creating the force data set to apply to the rod
        apply_force = np.zeros((3,self.n_elements+1))

        # PyElastica handles forces in GLOBAL coord. system, so they are applied directly. Also, the vertebra weights are added to each vertebra
        for i in range (len(self.vertebra_nodes)):
            apply_force[:,self.vertebra_nodes[i]] = self.force_data[i] + self.vertebra_weight_vector

        # Applies forces to the rod
        system.external_forces += apply_force


    def apply_torques(self, system: SystemType, time: np.float64 = 0.0):
        # The force_data set and vertebra_weight_vector are expressed in the global coordinate frame and must be changed to local reference frames for torque application
        # Creating the array which will contain the transformed force vectors
        transformed_force_data = np.zeros((len(self.vertebra_nodes), 3), dtype=np.float64)

        # Transforming the force vectors calculated in the compute_forces method from the global reference frame to the local reference frame
        for i in range(len(self.vertebra_nodes)):
            transformed_force_data[i] = system.director_collection[...,(self.vertebra_nodes[i]-1)] @ self.force_data[i]

        self.compute_torques(
            self.vertebra_height_vector, np.array(self.vertebra_nodes), transformed_force_data,
            self.n_elements, system.external_torques
        )


    @staticmethod
    @njit(cache=True)
    def get_rotations(position_collection, director_collection, vertebra_nodes, vertebra_height_vector):
        # Returns an array containing the unit norm vector which describes the orientation of each segment of tendon between vertebrae

        # Initializing unit_norm_vector_array to store the unit normed vectors that describe the global orientation of the forces in each vertebra
        unit_norm_vector_array = np.zeros((len(vertebra_nodes), 3), dtype=np.float64)

        for i in range(len(vertebra_nodes)+1):
            # There is a +1 in the for loop to account for the force between the first vertebra and the fixed node

            # If statement, used for the case when i = 0 and thus there is no vertebra before this one, same for the final vertebra (no vertebra after that one)
            if i==0:
                current_vertebra = 0
                next_vertebra = vertebra_nodes[i]
            elif i==len(vertebra_nodes):
                current_vertebra = vertebra_nodes[i-1]
                next_vertebra = vertebra_nodes[i-1]
            else:
                current_vertebra = vertebra_nodes[i-1]
                next_vertebra = vertebra_nodes[i]

            # Setting up values to be used iteratively
            x_current = position_collection[0, current_vertebra]
            y_current = position_collection[1, current_vertebra]
            z_current = position_collection[2, current_vertebra]

            x_next = position_collection[0, next_vertebra]
            y_next = position_collection[1, next_vertebra]
            z_next = position_collection[2, next_vertebra]

            current_rotation_matrix = director_collection[...,current_vertebra]
            next_rotation_matrix = director_collection[...,next_vertebra]

            current_node = np.array([x_current, y_current, z_current])
            next_node = np.array([x_next, y_next, z_next])

            # Calculating relative position vector between vertebrae, considering the vertebra height
            # Continguous arrays to help with computation speed
            delta_vector = (next_node + np.ascontiguousarray(next_rotation_matrix.T) @ np.ascontiguousarray(vertebra_height_vector)) - (current_node + np.ascontiguousarray(current_rotation_matrix.T) @ np.ascontiguousarray(vertebra_height_vector))

            # Calculating the unit-normed vector based on the differences calculated in the previous step
            delta_vector_norm = np.linalg.norm(delta_vector)
            unit_norm_delta_vector = delta_vector / delta_vector_norm

            # This if statement is to stop unit_norm_delta_vector from becoming a 'nan'
            if i==len(vertebra_nodes):
                unit_norm_delta_vector = np.zeros(3)

            # Storing the unit normed vector to be later used in the compute_forces method
            unit_norm_vector_array[i] = unit_norm_delta_vector

        return unit_norm_vector_array

    @staticmethod
    @njit(cache=True)
    def compute_forces(tension, vertebra_nodes, unit_norm_vector_array):

        # Creating array to store forces in vertebrae
        force_data = np.zeros((len(vertebra_nodes), 3), dtype=np.float64)

        for i in range(len(vertebra_nodes)):
            # This for loop multiplies the unit normed vectors calculated previously, with the tension of the tendon, thus creating the force vector for each vertebra
            # Contiguous array to increase speed in njit decorator
            force_current_prev = unit_norm_vector_array[i] * -tension
            force_current_next = unit_norm_vector_array[i+1] * tension

            # Summing the components of both force vectors to get the final force vector, which is then stored for use in the apply_forces and compute_torques methods
            force_data[i] = force_current_prev + force_current_next

        return force_data


    @staticmethod
    @njit(cache=True)
    def compute_torques(vertebra_height_vector, vertebra_nodes, transformed_force_data, n_elements, external_torques):

        # Creating torque data set for storage
        torque_data = np.zeros((len(vertebra_nodes), 3),dtype=np.float64)

        # Goes through vertebra nodes to calculate torques for them
        for i in range(len(vertebra_nodes)):

            # Cross product between the vertebra height vector and the local force vector due to the tendons, to obtain the tendon torque for that vertebra
            torque_vector = np.cross(vertebra_height_vector, transformed_force_data[i])

            # Sum of the vectors, and storage into the torque_data array
            torque_data[i] = torque_vector

        # Appending the computed torque vector to the final torque data set
        apply_torque = np.zeros((3,n_elements+1))

        k = 0
        for i in range(n_elements):
            if i in vertebra_nodes:
                apply_torque[:,i] = torque_data[k]
                k += 1
        apply_torque = apply_torque[:,1:]

        # Applying the torque data set to the rod (torque on the final vertebra)
        external_torques += apply_torque