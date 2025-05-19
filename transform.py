"""
Python Image Manipulation by Kylie Ying (modified from MIT 6.865)

YouTube Kylie Ying: https://www.youtube.com/ycubed 
Twitch KylieYing: https://www.twitch.tv/kylieying 
Twitter @kylieyying: https://twitter.com/kylieyying 
Instagram @kylieyying: https://www.instagram.com/kylieyying/ 
Website: https://www.kylieying.com
Github: https://www.github.com/kying18 
Programmer Beast Mode Spotify playlist: https://open.spotify.com/playlist/4Akns5EUb3gzmlXIdsJkPs?si=qGc4ubKRRYmPHAJAIrCxVQ 
"""
# Written by Kylie Ying and Developed by phoenix marie 
from PIL import Image as PILImage, ImageDraw, ImageFont
import numpy as np
import os

class Image:
    def __init__(self, x_pixels=None, y_pixels=None, num_channels=None, filename=None):
        if filename:
            self.pil_image = PILImage.open(filename).convert('RGB')
            self.array = np.array(self.pil_image) / 255.0
            self.x_pixels, self.y_pixels = self.array.shape[0], self.array.shape[1]
            self.num_channels = self.array.shape[2]
        else:
            self.x_pixels = x_pixels
            self.y_pixels = y_pixels
            self.num_channels = num_channels
            self.array = np.zeros((x_pixels, y_pixels, num_channels))

    def write_image(self, filename):
        im = np.clip(self.array * 255, 0, 255).astype(np.uint8)
        PILImage.fromarray(im).save(filename)

    def draw_rectangle(self, top_left, bottom_right, color=(1, 0, 0), thickness=2):
        img = (self.array * 255).astype(np.uint8)
        pil_img = PILImage.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        for i in range(thickness):
            draw.rectangle([top_left, bottom_right], outline=tuple(int(c * 255) for c in color))
        self.array = np.array(pil_img) / 255.0

    def add_text(self, text, position, color=(1, 1, 1), font_size=20):
        img = (self.array * 255).astype(np.uint8)
        pil_img = PILImage.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        draw.text(position, text, fill=tuple(int(c * 255) for c in color), font=font)
        self.array = np.array(pil_img) / 255.0

    def invert_colors(self):
        self.array = 1.0 - self.array

    def grayscale(self):
        gray = np.mean(self.array, axis=2)
        self.array = np.stack([gray, gray, gray], axis=2)

def brighten(image, factor):
    new_im = Image(x_pixels=image.x_pixels, y_pixels=image.y_pixels, num_channels=image.num_channels)
    new_im.array = np.clip(image.array * factor, 0, 1)
    return new_im

def adjust_contrast(image, factor, mid):
    new_im = Image(x_pixels=image.x_pixels, y_pixels=image.y_pixels, num_channels=image.num_channels)
    new_im.array = np.clip((image.array - mid) * factor + mid, 0, 1)
    return new_im

def blur(image, kernel_size):
    new_im = Image(x_pixels=image.x_pixels, y_pixels=image.y_pixels, num_channels=image.num_channels)
    neighbor_range = kernel_size // 2
    for x in range(image.x_pixels):
        for y in range(image.y_pixels):
            for c in range(image.num_channels):
                total = 0
                count = 0
                for x_i in range(max(0, x - neighbor_range), min(image.x_pixels, x + neighbor_range + 1)):
                    for y_i in range(max(0, y - neighbor_range), min(image.y_pixels, y + neighbor_range + 1)):
                        total += image.array[x_i, y_i, c]
                        count += 1
                new_im.array[x, y, c] = total / count
    return new_im

def apply_kernel(image, kernel):
    new_im = Image(x_pixels=image.x_pixels, y_pixels=image.y_pixels, num_channels=image.num_channels)
    neighbor_range = kernel.shape[0] // 2
    for x in range(image.x_pixels):
        for y in range(image.y_pixels):
            for c in range(image.num_channels):
                total = 0
                for x_i in range(-neighbor_range, neighbor_range + 1):
                    for y_i in range(-neighbor_range, neighbor_range + 1):
                        xi = min(max(x + x_i, 0), image.x_pixels - 1)
                        yi = min(max(y + y_i, 0), image.y_pixels - 1)
                        kernel_val = kernel[x_i + neighbor_range, y_i + neighbor_range]
                        total += image.array[xi, yi, c] * kernel_val
                new_im.array[x, y, c] = total
    return new_im

def combine_images(image1, image2):
    new_im = Image(x_pixels=image1.x_pixels, y_pixels=image1.y_pixels, num_channels=image1.num_channels)
    new_im.array = np.clip(np.sqrt(np.square(image1.array) + np.square(image2.array)), 0, 1)
    return new_im

def batch_process(input_dir, output_dir, func, *args):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = Image(filename=os.path.join(input_dir, filename))
            processed_img = func(img, *args)
            output_path = os.path.join(output_dir, filename)
            processed_img.write_image(output_path)

def compute_histogram(image):
    hist = {}
    for i, color in enumerate(['R', 'G', 'B']):
        hist[color] = np.histogram(image.array[:, :, i], bins=256, range=(0, 1))[0]
    return hist

def image_statistics(image):
    stats = {
        'mean_brightness': np.mean(image.array),
        'std_dev': np.std(image.array),
        'channel_means': {
            'R': np.mean(image.array[:, :, 0]),
            'G': np.mean(image.array[:, :, 1]),
            'B': np.mean(image.array[:, :, 2]),
        }
    }
    return stats

if __name__ == '__main__':
    lake = Image(filename='lake.png')
    city = Image(filename='city.png')

    brightened_im = brighten(lake, 1.7)
    brightened_im.write_image('brightened.png')

    darkened_im = brighten(lake, 0.3)
    darkened_im.write_image('darkened.png')

    incr_contrast = adjust_contrast(lake, 2, 0.5)
    incr_contrast.write_image('increased_contrast.png')

    decr_contrast = adjust_contrast(lake, 0.5, 0.5)
    decr_contrast.write_image('decreased_contrast.png')

    blur_3 = blur(city, 3)
    blur_3.write_image('blur_k3.png')

    blur_15 = blur(city, 15)
    blur_15.write_image('blur_k15.png')

    sobel_x = apply_kernel(city, np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]))
    sobel_x.write_image('edge_x.png')

    sobel_y = apply_kernel(city, np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]))
    sobel_y.write_image('edge_y.png')

    sobel_xy = combine_images(sobel_x, sobel_y)
    sobel_xy.write_image('edge_xy.png')

    hist = compute_histogram(lake)
    stats = image_statistics(lake)
    print("Histogram (R channel first 10 bins):", hist['R'][:10])
    print("Image statistics:", stats)

    lake.draw_rectangle((10, 10), (100, 100), color=(0, 1, 0))
    lake.add_text("Lake Image", position=(10, 110))
    lake.write_image("lake_annotated.png")

    lake.invert_colors()
    lake.write_image("lake_inverted.png")

    lake.grayscale()
    lake.write_image("lake_grayscale.png")
 
