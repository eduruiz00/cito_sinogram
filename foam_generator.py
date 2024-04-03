import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from collections import namedtuple


RADIUS_CYLINDER = 0.5
CENTER_CYLINDER = (0.5, 0.5)

class FoamGenerator:
    def __init__(self, img_pixels: int = 512, num_spheres: int = 1000, prob_overlap: float = 0):
        self.img_pixels = img_pixels
        self.num_spheres = num_spheres
        self.prob_overlap = prob_overlap

    def cylinder(self, im: npt.NDArray, center, radius: float, intensity: float = 1):
        assert len(im.shape) == 3, "Image must be 3D"
        xx, yy, zz = im.shape
        center_pixels = (int(round(center[0] * self.img_pixels)), int(round(center[1] * self.img_pixels)))
        radius_pixels = int(round(radius * self.img_pixels))
        x_coords, y_coords = np.meshgrid(np.arange(xx), np.arange(yy))
        dist_squared = (x_coords - center_pixels[0]) ** 2 + (y_coords - center_pixels[1]) ** 2
    
        mask = dist_squared <= radius_pixels ** 2
        for z in range(zz):
            im[:, :, z][mask] = intensity
        return im

    @staticmethod
    def inside_foam(foam_center, sphere_center):
        return (foam_center[0] - sphere_center[0])**2 + (foam_center[1] - sphere_center[1])**2 <= 0.5**2

    def distance_to_border(self, foam_center, foam_radius, sphere_center):
        distance_radius_foam = foam_radius - np.sqrt((foam_center[0] - sphere_center[0]) ** 2 + (foam_center[1] - sphere_center[1]) ** 2)
        distance_height = min(sphere_center[2], 1 - sphere_center[2])
        return np.min([distance_radius_foam, distance_height]) - 1 / self.img_pixels

    def distances_centers(self, buffer, sphere_center, overlap: bool = False):
        distance_border = self.distance_to_border(CENTER_CYLINDER, RADIUS_CYLINDER, sphere_center)
        if distance_border <= 0:
            return None
        distances = [distance_border]

        for s in buffer:
            dist_radius = np.sqrt((s.center_x - sphere_center[0]) ** 2 + (s.center_y - sphere_center[1]) ** 2 + (s.center_z - sphere_center[2]) ** 2)
            dist = max(dist_radius - s.radius, 0)
            if overlap:
                dist = dist * (1 + np.random.rand() * 3)
            if dist <= 0:
                return None
            distances.append(dist)
        return distances

    def print_sphere(self, im, center, radius):
        center_pixels = (int(round(center[0] * self.img_pixels)), int(round(center[1] * self.img_pixels)), int(round(center[2] * self.img_pixels)))
        radius_pixels = int(round(radius * self.img_pixels))
    
        for z in range(center_pixels[2] - radius_pixels, center_pixels[2] + radius_pixels + 1):
            for x in range(center_pixels[0] - radius_pixels, center_pixels[0] + radius_pixels + 1):
                for y in range(center_pixels[1] - radius_pixels, center_pixels[1] + radius_pixels + 1):
                    if (x - center_pixels[0])**2 + (y - center_pixels[1])**2 + (z - center_pixels[2])**2 <= radius_pixels**2:
                        im[x][y][z] = 0
        return im

    def run(self):
        image = np.zeros((self.img_pixels, self.img_pixels, self.img_pixels))
        sphere = namedtuple("sphere", ["center_x", "center_y", "center_z", "radius"])
    
        # Generate the big cylinder
        image = self.cylinder(image, CENTER_CYLINDER, RADIUS_CYLINDER, intensity=1)
    
        buffer = []
        generated_spheres = 0
    
        while generated_spheres < self.num_spheres:
            sphere_center = np.random.uniform(2 / self.img_pixels, 1 - 2 / self.img_pixels, 3)
            overlap = np.random.rand() < self.prob_overlap
    
            d = self.distances_centers(buffer, sphere_center, overlap)
            if d is None or not self.inside_foam(CENTER_CYLINDER, (sphere_center[0], sphere_center[1])):
                continue
    
            min_radius = 1 / self.img_pixels
            max_radius = min(min(d), RADIUS_CYLINDER * 0.2)
    
            sphere_radius = np.random.uniform(low=min_radius, high=max_radius)
    
            image = self.print_sphere(image, sphere_center, sphere_radius)
    
            sphere_ = sphere(sphere_center[0], sphere_center[1], sphere_center[2], sphere_radius)
            buffer.append(sphere_)
    
            generated_spheres += 1
            if generated_spheres % 1000 == 0:
                print(f"Generated {generated_spheres} / {self.num_spheres} spheres")
        return image


if __name__ == "__main__":
    generator = FoamGenerator()
    image = generator.run()
    image = image.transpose(2, 0, 1)
    fig = plt.figure()
    plt.subplot(2, 2, 1)
    plt.title("Slice 2")
    plt.imshow(image[2, :, :], cmap='gray')
    plt.subplot(2, 2, 2)
    plt.title(f"Slice {generator.img_pixels // 4}")
    plt.imshow(image[generator.img_pixels // 4, :, :], cmap='gray')
    plt.subplot(2, 2, 3)
    plt.title(f"Slice {generator.img_pixels // 2}")
    plt.imshow(image[generator.img_pixels // 2, :, :], cmap='gray')
    plt.subplot(2, 2, 4)
    plt.title(f"Slice {generator.img_pixels * 3 // 4}")
    plt.imshow(image[(3 * generator.img_pixels) // 4, :, :], cmap='gray')
    plt.savefig("phantom_diff_levels.png")
    plt.show()

    fig = plt.figure()
    cnt = 0
    for i in [-1, 0, 1, 2]:
        plt.subplot(2, 2, cnt + 1)
        plt.title(f"Slice {i + (generator.img_pixels // 2)}")
        plt.imshow(image[i + (generator.img_pixels // 2), :, :], cmap='gray')
        cnt += 1
    plt.savefig("phantom_seq.png")
    plt.show()
