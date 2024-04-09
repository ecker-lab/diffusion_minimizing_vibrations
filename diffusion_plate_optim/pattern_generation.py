import math, torch
from torch.distributions import Uniform
from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops


def rot_mat(theta):
    theta_rad = theta * math.pi / 180
    matrix = [
        [math.cos(theta_rad), -math.sin(theta_rad)],
        [math.sin(theta_rad),  math.cos(theta_rad)]
    ]
    return matrix


def draw_line(draw, max_length, resolution, w_lines, length="sampled"):
    angle = Uniform(0, 180).sample()
    if length == "sampled":
        length = Uniform(0, max_length).sample()
    else:
        length = max_length
    width = int(np.random.uniform(w_lines[0], w_lines[1]) * resolution[0] / 100)
    line = np.array([((-length/2, 0.0), (length/2, 0.0))]) @ rot_mat(angle)

    delta_x = np.random.uniform(resolution[0])
    line[:,:,0] = line[:,:,0] + delta_x
    delta_y = np.random.uniform(0, resolution[1])
    line[:,:,1] = line[:,:,1] + delta_y

    draw.line((line[0,0,0], line[0,0,1], line[0,1,0], line[0,1,1]), fill='white', width=width)
    return draw


def draw_ellipse(draw, img, w_ellipses, resolution, length_ellipse):
    width1, width2, width3 = np.random.uniform(w_ellipses[0], w_ellipses[1], 3) * resolution[0] / 100
    x_mid = resolution[0] + np.random.uniform(0, resolution[0])
    y_mid = resolution[1] + np.random.uniform(0, resolution[1])
    length_x, length_y = np.random.uniform(length_ellipse[1]* resolution[0] / 100, resolution[0] - length_ellipse[0]* resolution[0] / 100, 2)

    img_ellipse = Image.new('L', (3*resolution[0], 3*resolution[1]), color=(0))
    draw_ellipse_element = ImageDraw.Draw(img_ellipse)

    x = np.array([x_mid - length_x/2, y_mid - length_y/2])
    y = np.array([x_mid + length_x/2, y_mid + length_y/2])
    xy = [x[0], x[1], y[0], y[1]]

    draw_ellipse_element.ellipse(xy, fill = "white", outline ="white", width=width3)

    x2 = x + width1
    y2 = y - width2
    draw_ellipse_element.ellipse([x2[0], x2[1], y2[0], y2[1]], fill = "black", outline ="black")

    angle = np.random.uniform(0, 90, 1)
    img_ellipse = img_ellipse.rotate(angle)
    img_ellipse = img_ellipse.crop((resolution[0], resolution[1], 2*resolution[0], 2*resolution[1]))
    img = ImageChops.lighter(img, img_ellipse)
    return draw, img


def draw_bounding_box(draw, resolution, width):
    fill_color = 'black'
    p1 = (0,0)
    p2 = (0,resolution[1])
    p3 = (resolution[0], resolution[1])
    p4 = (resolution[0], 0)
    p5 = p1 + width
    p6 = (0 + width,resolution[1] - width)
    p7 = p3 - width
    p8 = (resolution[0] - width, 0 + width)
    left_box = np.array([p1, p2, p6, p5])
    upper_box = np.array([p2, p3, p7, p6])
    right_box = np.array([p3, p4, p8, p7])
    lower_box = np.array([p1, p5, p8, p4])
    boxes = [left_box, upper_box, right_box, lower_box]

    for box in boxes:
        draw.polygon([tuple(p) for p in box], fill=fill_color, width=0, outline=fill_color)
    return draw


def draw_simple_img(resolution=[121, 81], n_lines=[1, 2], w_lines=[4, 7], n_ellipses=[0, 2], w_ellipses=[3, 5], gauss_blur=[1.05, 1.45], length_ellipse=[10, 15]):
    img = Image.new('L', (resolution[0], resolution[1]), color=(0))
    draw = ImageDraw.Draw(img)
    max_length = np.sqrt(np.square(resolution[0]) + np.square(resolution[1])) 
    n_lines_plot = np.random.randint(n_lines[0], n_lines[1] + 1)
    for i in range(n_lines_plot):
        draw = draw_line(draw, max_length, resolution, w_lines, length="fixed")

    n_ellipses_plot = np.random.randint(n_ellipses[0], n_ellipses[1] + 1)
    for i in range(n_ellipses_plot):
        draw, img = draw_ellipse(draw, img, w_ellipses, resolution, length_ellipse)
    draw_bounding_box(draw, resolution, width=np.array(5 * resolution[0] / 100))
    img = img.filter(ImageFilter.GaussianBlur(np.random.uniform(*gauss_blur)))

    img_np = np.array(img).flatten()
    img_np = img_np[img_np > 0]
    if len(img_np) > 0.8 * resolution[0] * resolution[1]:
        img = draw_simple_img()

    return img


class PlateDataset(Dataset):
    def __init__(self, dimension=np.array([0.9, 0.6]), resolution=[121, 81], n_lines=[1, 4], w_lines=[20, 70], n_ellipses=[1, 4], w_ellipses=[20, 50], gauss_blur=[1.05, 1.45], length_ellipse=[30, 150]):
        self.dimension = dimension
        self.resolution = resolution
        self.n_lines = n_lines
        self.w_lines = w_lines
        self.n_ellipses = n_ellipses
        self.w_ellipses = w_ellipses
        self.gauss_blur = gauss_blur
        self.length_ellipse = length_ellipse
        self.n_samples = 20000

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        img = draw_simple_img()
        return {"bead_patterns": torch.from_numpy(np.array(img)).float().unsqueeze(0)/255}
