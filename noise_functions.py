import numpy as np
import cv2
import random


def add_gaussian_noise(image, mean=0, std=25):
    """ 添加高斯噪声到图像中。 """
    gaussian_noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, gaussian_noise)
    return noisy_image


def add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    """ 添加椒盐噪声到图像中。 """
    noisy_image = np.copy(image)
    total_pixels = image.size

    # 添加盐噪声
    num_salt = np.ceil(salt_prob * total_pixels).astype(int)
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 255  # 假设图像是灰度图，255为白色

    # 添加胡椒噪声
    num_pepper = np.ceil(pepper_prob * total_pixels).astype(int)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 0  # 假设图像是灰度图，0为黑色

    return noisy_image


def add_poisson_noise(image):
    """ 添加泊松噪声到图像中。 """
    noisy_image = np.random.poisson(image).astype(np.uint8)
    return noisy_image


def add_speckle_noise(image):
    """ 添加散斑噪声到图像中。 """
    gauss = np.random.normal(0, 0.1, image.shape).astype(np.uint8)
    noisy_image = image + image * gauss
    return noisy_image


def add_compression_noise(image, quality=10):
    """ 添加压缩噪声到图像中。 """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, compressed_image = cv2.imencode('.jpg', image, encode_param)
    noisy_image = cv2.imdecode(compressed_image, 1)
    return noisy_image


def add_blur_noise(image, kernel_size=5):
    """ 添加模糊噪声到图像中。 """
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    blurred_image = cv2.filter2D(image, -1, kernel)
    return blurred_image


# Example usage:
if __name__ == "__main__":
    image = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

    gaussian_noisy = add_gaussian_noise(image)
    salt_pepper_noisy = add_salt_and_pepper_noise(image)
    poisson_noisy = add_poisson_noise(image)
    speckle_noisy = add_speckle_noise(image)
    compression_noisy = add_compression_noise(image)
    blur_noisy = add_blur_noise(image)

    # Display or save the noisy images as needed
    cv2.imshow("Gaussian Noise", gaussian_noisy)
    cv2.imshow("Salt and Pepper Noise", salt_pepper_noisy)
    cv2.imshow("Poisson Noise", poisson_noisy)
    cv2.imshow("Speckle Noise", speckle_noisy)
    cv2.imshow("Compression Noise", compression_noisy)
    cv2.imshow("Blur Noise", blur_noisy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
