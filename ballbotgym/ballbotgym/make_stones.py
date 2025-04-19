import numpy as np
from scipy.spatial import ConvexHull
import trimesh
from noise import pnoise3
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def generate_stone(num_points=500, noise_scale=0.5, noise_freq=3.0, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Sample points roughly inside a sphere
    points = np.random.normal(size=(num_points, 3))
    points /= np.linalg.norm(points, axis=1)[:, None]
    points *= np.random.uniform(0.5, 1.0, size=(num_points, 1))

    # Add Perlin noise
    def noisy(p):
        return p + noise_scale * np.array([
            pnoise3(*p * noise_freq),
            pnoise3(*(p * noise_freq + 10)),
            pnoise3(*(p * noise_freq + 20))
        ])

    noisy_points = np.array([noisy(p) for p in points])

    # Build convex hull
    hull = ConvexHull(noisy_points)
    vertices = noisy_points
    faces = hull.simplices

    return vertices, faces

def display_mesh_matplotlib(vertices, faces):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Build face list for plotting
    mesh_faces = [[vertices[idx] for idx in face] for face in faces]
    collection = Poly3DCollection(mesh_faces, alpha=0.7, edgecolor='k')
    ax.add_collection3d(collection)

    scale = np.ptp(vertices, axis=0)
    center = np.mean(vertices, axis=0)
    radius = max(scale) * 0.6

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect([1, 1, 1])

    plt.tight_layout()
    plt.show()

def save_mesh_as_stl(vertices, faces, filename='stone.stl'):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.export(filename)
    print(f"Saved to: {filename}")

if __name__ == "__main__":
    verts, tris = generate_stone()
    display_mesh_matplotlib(verts, tris)
    save_mesh_as_stl(verts, tris, "random_stone.stl")

