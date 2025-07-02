import numpy as np

def calculate_centroid(atom_positions, weights=None):
    """
    Calculates the centroid of the atom positions.

    Parameters:
    - atom_positions: A (n, 3) numpy array containing the positions of n atoms.
    - weights: Optional numpy array of shape (n,) containing weights for each atom.

    Returns:
    - centroid: The centroid of the atom positions.
    """
    if weights is None:
        weights = np.ones(atom_positions.shape[0])
    centroid = np.average(atom_positions, axis=0, weights=weights)
    return centroid

def center_positions(atom_positions, centroid):
    """
    Centers the atom positions around the centroid.

    Parameters:
    - atom_positions: A (n, 3) numpy array containing the positions of n atoms.
    - centroid: The centroid of the atom positions.

    Returns:
    - centered_positions: The atom positions centered around the centroid.
    """
    centered_positions = atom_positions - centroid
    return centered_positions

def calculate_covariance_matrix(centered_positions, weights=None):
    """
    Calculates the covariance matrix of the centered atom positions.

    Parameters:
    - centered_positions: The atom positions centered around the centroid.
    - weights: Optional numpy array of shape (n,) containing weights for each atom.

    Returns:
    - cov_matrix: The covariance matrix.
    """
    if weights is None:
        cov_matrix = np.cov(centered_positions, rowvar=False)
    else:
        cov_matrix = np.cov(centered_positions, aweights=weights, rowvar=False)
    return cov_matrix

def fit_plane(cov_matrix):
    """
    Fits a plane through the atom positions based on the covariance matrix.

    Parameters:
    - cov_matrix: The covariance matrix of the centered atom positions.

    Returns:
    - normal_vector: The normal vector to the fitted plane.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    normal_vector = eigenvectors[:, 0]
    return normal_vector

def calculate_distances_from_plane(centered_positions, normal_vector):
    """
    Calculates the distances of each atom from the fitted plane.

    Parameters:
    - centered_positions: The atom positions centered around the centroid.
    - normal_vector: The normal vector to the fitted plane.

    Returns:
    - distances: The distances of each atom from the fitted plane.
    """
    distances = np.abs(np.dot(centered_positions, normal_vector))
    return distances

def plane_penalty(atom_positions, weights=None):
    """
    Calculates the penalty (negative reward) as the sum of distances from the fitted plane.

    Parameters:
    - atom_positions: A (n, 3) numpy array containing the positions of n atoms.
    - weights: Optional numpy array of shape (n,) containing weights for each atom.
               If None is provided, weights are set to ones for all atoms.

    Returns:
    - penalty: The penalty (negative reward).
    """
    if weights is None:
        weights = np.ones(atom_positions.shape[0])
    
    centroid = calculate_centroid(atom_positions, weights)
    centered_positions = center_positions(atom_positions, centroid)
    cov_matrix = calculate_covariance_matrix(centered_positions, weights)
    normal_vector = fit_plane(cov_matrix)
    distances = calculate_distances_from_plane(centered_positions, normal_vector)
    penalty = np.sum(distances * weights)
    return -penalty  # Return negative penalty as a reward


def center_of_mass_penalty(atom_positions, weights=None):
    """
    Calculates the distance of the atom positions from the center of mass.

    Parameters:
    - atom_positions: A (n, 3) numpy array containing the positions of n atoms.
    - weights: Optional numpy array of shape (n,) containing weights for each atom.
               If None is provided, weights are set to ones for all atoms.

    Returns:
    - distance_penalty: The penalty (negative reward) based on the distance from the center of mass.
    """
    if weights is None:
        weights = np.ones(atom_positions.shape[0])
    
    centroid = calculate_centroid(atom_positions, weights)
    distances = np.linalg.norm(atom_positions - centroid, axis=1)
    
    # Calculate the penalty as the weighted sum of distances
    distance_penalty = np.sum(distances * weights)
    
    return -distance_penalty  # Return negative penalty as a reward


def get_max_view_positions(atom_positions):

    weights = np.ones(len(atom_positions))
    centroid = calculate_centroid(atom_positions, weights)
    centered_positions = center_positions(atom_positions, centroid)
    cov_matrix = calculate_covariance_matrix(centered_positions, weights)
    normal_vector = fit_plane(cov_matrix)

    # Normalize the normal vector
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    z_axis = np.array([0, 0, 1])

    # Compute rotation axis (cross product) and angle
    rotation_axis = np.cross(normal_vector, z_axis)
    sin_angle = np.linalg.norm(rotation_axis)
    cos_angle = np.dot(normal_vector, z_axis)

    if sin_angle < 1e-8:
        # Already aligned or opposite
        if cos_angle > 0:
            rotation_matrix = np.eye(3)  # Already aligned
        else:
            # Opposite direction: rotate 180 degrees around any orthogonal axis
            # Find an orthogonal vector
            orthogonal = np.array([1, 0, 0]) if abs(normal_vector[0]) < 0.9 else np.array([0, 1, 0])
            rotation_axis = np.cross(normal_vector, orthogonal)
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            angle = np.pi
            # Rodrigues' rotation formula
            K = np.array([
                [0, -rotation_axis[2], rotation_axis[1]],
                [rotation_axis[2], 0, -rotation_axis[0]],
                [-rotation_axis[1], rotation_axis[0], 0]
            ])
            rotation_matrix = (
                np.eye(3) +
                np.sin(angle) * K +
                (1 - np.cos(angle)) * K @ K
            )
    else:
        # General case: use Rodrigues' rotation formula
        rotation_axis = rotation_axis / sin_angle
        angle = np.arctan2(sin_angle, cos_angle)

        K = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ])
        rotation_matrix = (
            np.eye(3) +
            np.sin(angle) * K +
            (1 - np.cos(angle)) * K @ K
        )
        
    # Add a small extra rotation 
    rot_size = 1 # radians
    extra_rotation = np.array([
        [np.cos(rot_size), 0, np.sin(rot_size)],
        [0, 1, 0], 
        [-np.sin(rot_size), 0, np.cos(rot_size)]
    ])
    rotation_matrix = rotation_matrix @ extra_rotation

    # Apply the rotation to the atomic positions
    rotated_positions = atom_positions @ rotation_matrix.T

    return rotated_positions
