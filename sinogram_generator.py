import astra
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import time

from foam_generator import FoamGenerator


NUM_PROJECTIONS = 1024
NUM_ITERATIONS = 200


def generate_foam(num_pixels: int = 256, num_spheres: int = 1000, prob_overlap: float = 0.0) -> npt.NDArray:
    """
    Generate a foam phantom.
    :param num_pixels: The number of pixels in the image.
    :param num_spheres: The number of spheres in the foam.
    :param prob_overlap: The probability of overlap between spheres.
    :return: The foam phantom.
    """
    print("Generating foam phantom...")
    time_foam = time.time()
    generator = FoamGenerator(img_pixels=num_pixels, num_spheres=num_spheres, prob_overlap=prob_overlap)
    image_foam = generator.run()
    image_foam = image_foam.transpose(2, 0, 1)
    time_foam = (time.time() - time_foam)
    print(f"Finished foam, time: {time_foam}")
    return image_foam


def sinogram_with_noise(sinogram: npt.NDArray, photon_count: int) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Generate a noisy sinogram from a sinogram with Poisson noise.
    :param sinogram: The sinogram to add noise to.
    :param photon_count: The number of photons to simulate.
    :return: The noisy sinogram and the generated noise.
    """
    noisy_sinogram = np.random.poisson(photon_count * np.exp(-sinogram))
    noisy_sinogram[noisy_sinogram == 0] = 1
    noisy_sinogram = -np.log(noisy_sinogram / photon_count)
    generated_noise = noisy_sinogram - sinogram
    return noisy_sinogram, generated_noise


def calculate_absorption(sinogram: npt.NDArray) -> float:
    """
    Calculate the absorption of a sinogram.
    :param sinogram: The sinogram to calculate the absorption of.
    :return: The absorption of the sinogram.
    """
    return 1 - np.mean(np.exp(-sinogram[sinogram > 0]))


def calculate_attenuation(sinogram: npt.NDArray, absorption: float) -> float:
    """
    Calculate the attenuation of a sinogram.
    :param sinogram: The sinogram to calculate the attenuation of.
    :param absorption: The absorption of the sinogram.
    :return: The attenuation of the sinogram.
    """
    return (np.log(1 - absorption + 1e-6) / np.mean(-sinogram[sinogram != 0]))


def generate_noisy_sinogram(sinogram: npt.NDArray, absorption: float, photon_count: int) -> npt.NDArray:
    """
    Generate a noisy sinogram from a sinogram with Poisson noise and attenuation.
    :param sinogram: The sinogram to add noise to.
    :param absorption: The absorption of the sinogram.
    :param photon_count: The number of photons to simulate.
    :return: The noisy sinogram.
    """
    atten = calculate_attenuation(sinogram, absorption)
    noisy_sinogram = sinogram * atten
    print(f"Absorption: {round(calculate_absorption(noisy_sinogram)*100):0.0f} %")
    noisy_sinogram, _ = sinogram_with_noise(noisy_sinogram, photon_count)
    noisy_sinogram /= atten
    return noisy_sinogram


if __name__ == "__main__":
    # Import the phantom
    phantom = generate_foam()
    print("Phantom shape:", phantom.shape)

    # Generate the projection data
    angles = np.linspace(0, np.pi, NUM_PROJECTIONS,False)
    max_distance = int(np.sqrt(2) * phantom.shape[1]) + 1
    vol_geom = astra.create_vol_geom(phantom.shape)
    proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, phantom.shape[0], max_distance, angles)
    proj_id, proj_data = astra.create_sino3d_gpu(phantom, proj_geom, vol_geom)
    proj_data = proj_data.astype(np.float32)
    print("Projection generated.")
    print("Projection data shape:", proj_data.shape)


    # Plot 4 sinograms
    projections = [proj_data]
    titles = ["Original", "Noisy α=50%, I0=100", "Noisy α=50%, I0=1000", "Noisy α=80%, I0=100"]
    plt.figure()
    n_slices = proj_data.shape[0]
    plt.subplot(2, 2, 1)
    plt.imshow(proj_data[n_slices // 2], cmap='gray')
    plt.axis('off')
    plt.title("Original sinogram")
    plt.subplot(2, 2, 2)
    noisy_sinogram = generate_noisy_sinogram(proj_data, 0.5, 100)
    projections.append(noisy_sinogram)
    plt.imshow(noisy_sinogram[n_slices // 2], cmap='gray')
    plt.axis('off')
    plt.title("Noisy sinogram α=50%, I0=100")
    plt.subplot(2, 2, 3)
    noisy_sinogram = generate_noisy_sinogram(proj_data, 0.5, 1000)
    projections.append(noisy_sinogram)
    plt.imshow(noisy_sinogram[n_slices // 2], cmap='gray')
    plt.axis('off')
    plt.title("Noisy sinogram α=50%, I0=1000")
    plt.subplot(2, 2, 4)
    noisy_sinogram = generate_noisy_sinogram(proj_data, 0.8, 100)
    projections.append(noisy_sinogram)
    plt.imshow(noisy_sinogram[n_slices // 2], cmap='gray')
    plt.axis('off')
    plt.title("Noisy sinogram α=80%, I0=100")
    plt.suptitle("Sinograms")
    plt.savefig("noisy_sinograms.png", dpi=600)

    # Generate the reconstructions
    plt.figure()
    for i, proj in enumerate(projections):
        plt.subplot(2, 2, i + 1)
        proj_sino = astra.create_proj_geom('parallel', 1.0, max_distance, np.linspace(0, np.pi, NUM_PROJECTIONS, False))
        sinogram_id = astra.data2d.create('-sino', proj_sino, proj[proj.shape[0] // 2])
        recon_id = astra.data2d.create('-vol', vol_geom, 0)
        cfg = astra.astra_dict('SIRT_CUDA')
        cfg['ReconstructionDataId'] = recon_id
        cfg['ProjectionDataId'] = sinogram_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, NUM_ITERATIONS)
        rec = astra.data2d.get(recon_id)
        plt.imshow(rec, cmap='gray')
        plt.axis('off')
        plt.title(titles[i])
        print("Finished reconstruction", i)
    plt.suptitle("Reconstructions")
    plt.savefig("reconstructions.png", dpi=600)

    astra.algorithm.delete(alg_id)
    astra.data2d.delete(recon_id)
    astra.data2d.delete(sinogram_id)
    astra.projector.delete(proj_id)
