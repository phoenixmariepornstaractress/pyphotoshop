"""
Python Image Representation (modified from MIT 6.865)

YouTube Kylie Ying: https://www.youtube.com/ycubed 
Twitch KylieYing: https://www.twitch.tv/kylieying 
Twitter @kylieyying: https://twitter.com/kylieyying 
Instagram @kylieyying: https://www.instagram.com/kylieyying/ 
Website: https://www.kylieying.com
Github: https://www.github.com/kying18 
Programmer Beast Mode Spotify playlist: https://open.spotify.com/playlist/4Akns5EUb3gzmlXIdsJkPs?si=qGc4ubKRRYmPHAJAIrCxVQ 
"""
import numpy as np
import os
import png
from PIL import Image as PILImage, ImageFilter, ImageEnhance, ImageOps, ImageDraw, ImageFont

class Image:
    def __init__(self, x_pixels=0, y_pixels=0, num_channels=0, filename=''):
        self.input_path = 'input/'
        self.output_path = 'output/'
        os.makedirs(self.input_path, exist_ok=True)
        os.makedirs(self.output_path, exist_ok=True)

        self.history = []  # to track changes for undo

        if filename:
            self.array = self.read_image(filename)
            self.x_pixels, self.y_pixels, self.num_channels = self.array.shape
        elif x_pixels and y_pixels and num_channels:
            self.x_pixels, self.y_pixels, self.num_channels = x_pixels, y_pixels, num_channels
            self.array = np.zeros((x_pixels, y_pixels, num_channels), dtype=np.float32)
        else:
            raise ValueError("You must provide either a filename or image dimensions.")

    def _save_history(self):
        self.history.append(self.array.copy())

    def undo(self):
        if self.history:
            self.array = self.history.pop()

    def read_image(self, filename, gamma=2.2):
        reader = png.Reader(self.input_path + filename).asFloat()
        img_data = np.vstack(list(reader[2]))
        img_data.resize(reader[1], reader[0], 3)
        return img_data ** gamma

    def write_image(self, output_file_name, gamma=2.2):
        clipped = np.clip(self.array, 0, 1)
        y, x = clipped.shape[:2]
        reshaped = (clipped ** (1 / gamma)).reshape(y, x * 3)
        writer = png.Writer(x, y)
        with open(self.output_path + output_file_name, 'wb') as f:
            writer.write(f, (reshaped * 255).astype(np.uint8))
        self.array = clipped.reshape(y, x, 3)

    def _to_pil(self):
        return PILImage.fromarray((np.clip(self.array, 0, 1) * 255).astype(np.uint8), mode='RGB')

    def _from_pil(self, pil_img):
        return np.asarray(pil_img).astype(np.float32) / 255.0

    def batch_process(self, process_func, ext='png'):
        for file in os.listdir(self.input_path):
            if file.endswith(ext):
                img = Image(filename=file)
                process_func(img)
                img.write_image(f'processed_{file}')

    def invert(self):
        self._save_history()
        self.array = 1.0 - self.array

    def grayscale(self):
        self._save_history()
        if self.num_channels != 3:
            raise ValueError("Grayscale conversion supports only RGB images.")
        luminance = np.dot(self.array[..., :3], [0.299, 0.587, 0.114])
        self.array = np.repeat(luminance[:, :, np.newaxis], 3, axis=2)

    def threshold(self, thresh=0.5):
        self._save_history()
        self.grayscale()
        binary = (self.array[..., 0] > thresh).astype(np.float32)
        self.array = np.repeat(binary[:, :, np.newaxis], 3, axis=2)

    def resize(self, new_x, new_y):
        self._save_history()
        resized = self._to_pil().resize((new_y, new_x), resample=PILImage.LANCZOS)
        self.array = self._from_pil(resized)
        self.x_pixels, self.y_pixels = new_x, new_y

    def rotate(self, degrees):
        self._save_history()
        rotated = self._to_pil().rotate(degrees, expand=True)
        self.array = self._from_pil(rotated)
        self.x_pixels, self.y_pixels = self.array.shape[:2]

    def flip(self, direction='horizontal'):
        self._save_history()
        if direction == 'horizontal':
            self.array = np.flip(self.array, axis=1)
        elif direction == 'vertical':
            self.array = np.flip(self.array, axis=0)
        else:
            raise ValueError("Direction must be 'horizontal' or 'vertical'.")

    def crop(self, x1, y1, x2, y2):
        self._save_history()
        self.array = self.array[x1:x2, y1:y2, :]
        self.x_pixels, self.y_pixels = self.array.shape[:2]

    def adjust_brightness(self, factor):
        self._save_history()
        self.array = np.clip(self.array * factor, 0, 1)

    def apply_pil_filter(self, filter_type):
        self._save_history()
        filtered = self._to_pil().filter(filter_type)
        self.array = self._from_pil(filtered)

    def enhance(self, kind='contrast', factor=1.0):
        self._save_history()
        enhancer_classes = {
            'contrast': ImageEnhance.Contrast,
            'color': ImageEnhance.Color,
            'brightness': ImageEnhance.Brightness,
            'sharpness': ImageEnhance.Sharpness
        }
        enhancer = enhancer_classes.get(kind, ImageEnhance.Contrast)(self._to_pil())
        self.array = self._from_pil(enhancer.enhance(factor))

    def border(self, size, color=(0, 0, 0)):
        self._save_history()
        bordered = ImageOps.expand(self._to_pil(), border=size, fill=color)
        self.array = self._from_pil(bordered)
        self.x_pixels, self.y_pixels = self.array.shape[:2]

    def add_text(self, text, position=(10, 10), font_size=20, color=(255, 255, 255), max_width=None):
        self._save_history()
        img = self._to_pil()
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        if max_width:
            words = text.split()
            lines = []
            current_line = ""
            for word in words:
                if draw.textlength(current_line + " " + word, font=font) <= max_width:
                    current_line += (" " if current_line else "") + word
                else:
                    lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)
            y_offset = position[1]
            for line in lines:
                draw.text((position[0], y_offset), line, font=font, fill=color)
                y_offset += font.getsize(line)[1]
        else:
            draw.text(position, text, font=font, fill=color)

        self.array = self._from_pil(img)

    def draw_shape(self, shape='rectangle', coords=(0, 0, 50, 50), color=(255, 0, 0), width=3):
        self._save_history()
        img = self._to_pil()
        draw = ImageDraw.Draw(img)
        if shape == 'rectangle':
            draw.rectangle(coords, outline=color, width=width)
        elif shape == 'ellipse':
            draw.ellipse(coords, outline=color, width=width)
        elif shape == 'line':
            draw.line(coords, fill=color, width=width)
        elif shape == 'polygon':
            draw.polygon(coords, outline=color, width=width)
        self.array = self._from_pil(img)

    def save_as_format(self, output_filename, fmt='JPEG'):
        self._to_pil().save(self.output_path + output_filename, format=fmt)

    def gaussian_blur(self, radius=2):
        self._save_history()
        blurred = self._to_pil().filter(ImageFilter.GaussianBlur(radius))
        self.array = self._from_pil(blurred)

    def sepia_tone(self):
        self._save_history()
        img = self._to_pil().convert("RGB")
        sepia = np.array(img).astype(np.float32)
        r, g, b = sepia[:,:,0], sepia[:,:,1], sepia[:,:,2]
        tr = 0.393 * r + 0.769 * g + 0.189 * b
        tg = 0.349 * r + 0.686 * g + 0.168 * b
        tb = 0.272 * r + 0.534 * g + 0.131 * b
        sepia = np.stack([tr, tg, tb], axis=2)
        sepia = np.clip(sepia, 0, 255)
        self.array = sepia.astype(np.uint8).astype(np.float32) / 255.0

    def vignette(self):
        self._save_history()
        rows, cols = self.array.shape[:2]
        y, x = np.ogrid[:rows, :cols]
        y_center, x_center = rows / 2, cols / 2
        dist = np.sqrt((x - x_center)**2 + (y - y_center)**2)
        max_dist = np.sqrt(x_center**2 + y_center**2)
        mask = 1 - (dist / max_dist)
        mask = np.clip(mask, 0.3, 1)
        self.array = self.array * mask[..., np.newaxis]

    def add_noise(self, noise_type='gaussian', mean=0.0, std=0.05):
        self._save_history()
        if noise_type == 'gaussian':
            noise = np.random.normal(mean, std, self.array.shape)
            self.array = np.clip(self.array + noise, 0, 1)
        elif noise_type == 'salt_pepper':
            amount = 0.02
            salt_pepper = np.random.rand(*self.array.shape[:2])
            self.array[salt_pepper < amount] = 1
            self.array[salt_pepper > 1 - amount] = 0

    def perspective_transform(self, coeffs):
        self._save_history()
        pil_img = self._to_pil().transform(self._to_pil().size, PILImage.PERSPECTIVE, coeffs, PILImage.BICUBIC)
        self.array = self._from_pil(pil_img)

    def affine_transform(self, matrix):
        self._save_history()
        pil_img = self._to_pil().transform(self._to_pil().size, PILImage.AFFINE, matrix, PILImage.BICUBIC)
        self.array = self._from_pil(pil_img)

    def overlay_image(self, overlay_path, position=(0, 0), alpha=0.5):
        self._save_history()
        base = self._to_pil().convert("RGBA")
        overlay = PILImage.open(overlay_path).convert("RGBA").resize(base.size)
        blended = PILImage.blend(base, overlay, alpha)
        self.array = self._from_pil(blended.convert("RGB"))

if __name__ == '__main__':
    def sample_process(img):
        img.invert()
        img.resize(64, 64)
        img.add_text("Batch", position=(5, 5))

    Image().batch_process(sample_process)

    im = Image(filename='lake.png')
    im.invert()
    im.grayscale()
    im.threshold(0.4)
    im.resize(100, 100)
    im.rotate(90)
    im.flip('vertical')
    im.crop(10, 10, 90, 90)
    im.adjust_brightness(1.3)
    im.apply_pil_filter(ImageFilter.EDGE_ENHANCE)
    im.enhance('contrast', 1.5)
    im.enhance('color', 1.2)
    im.border(5, color=(255, 0, 0))
    im.add_text("Sample text with wrapping support", position=(5, 5), font_size=18, color=(255, 255, 0), max_width=90)
    im.draw_shape(shape='rectangle', coords=(5, 5, 60, 60), color=(0, 255, 0), width=2)
    im.gaussian_blur(radius=3)
    im.sepia_tone()
    im.vignette()
    im.add_noise(noise_type='gaussian')
    im.perspective_transform([1, 0.2, -30, 0.1, 1, -20, 0.0005, 0.0008])
    im.affine_transform([1, 0.3, 0, 0.1, 1, 0])
    im.overlay_image('input/watermark.png', alpha=0.3)
    im.write_image('test_output.png')
    im.save_as_format('test_output.jpg', fmt='JPEG')
    im.save_as_format('test_output.bmp', fmt='BMP')
    im.save_as_format('test_output.tiff', fmt='TIFF')
 
