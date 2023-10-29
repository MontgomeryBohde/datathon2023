import numpy as np
import cairocffi as cairo
import matplotlib.pyplot as plt
import json
from simplification.cutil import simplify_coords
import struct
from struct import unpack
from sklearn.preprocessing import OneHotEncoder
from skimage import io, transform
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
import os
import pickle
import tqdm
import random


def stroke_to_points(stroke):
    return [[x, y] for x, y in zip(stroke[0], stroke[1])]

def points_to_stroke(points):
    return [[x for x, y in points], [y for x, y in points]]

def min_drawing(drawing):
    min_x = min([min(stroke[0]) for stroke in drawing])
    min_y = min([min(stroke[1]) for stroke in drawing])
    
    return min_x, min_y

def max_drawing(drawing):
    max_x = max([max(stroke[0]) for stroke in drawing])
    max_y = max([max(stroke[1]) for stroke in drawing])
    
    return max_x, max_y

def shift(drawing):
    min_x, min_y = min_drawing(drawing)

    return [[[x - min_x for x in stroke[0]], [y - min_y for y in stroke[1]]] for stroke in drawing]

def scale(drawing):
    max_value = max(max_drawing(drawing))
    if max_value == 0:
        return drawing
    return [[[x / max_value * 255 for x in stroke[0]], [y / max_value * 255 for y in stroke[1]]] for stroke in drawing]

def rdp_simplify(drawing):
    return [points_to_stroke(simplify_coords(stroke_to_points(stroke), 2.0)) for stroke in drawing]

def vector_to_raster(vector_images, side=28, line_diameter=10, padding=10, bg_color=(0,0,0), fg_color=(1,1,1)):
    """
    padding and line_diameter are relative to the original 256x256 image.
    """
    
    original_side = 256.0
    
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, side, side)
    ctx = cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_width(line_diameter)

    # scale to match the new size
    # add padding at the edges for the line_diameter
    # and add additional padding to account for antialiasing
    total_padding = padding * 2. + line_diameter
    new_scale = float(side) / float(original_side + total_padding)
    ctx.scale(new_scale, new_scale)
    ctx.translate(total_padding / 2., total_padding / 2.)

    raster_images = []
    for vector_image in vector_images:
        # clear background
        ctx.set_source_rgb(*bg_color)
        ctx.paint()
        bbox = np.hstack(vector_image).max(axis=1)
        offset = ((original_side, original_side) - bbox) / 2.
        offset = offset.reshape(-1,1)
        centered = [stroke + offset for stroke in vector_image]

        # draw strokes, this is the most cpu-intensive part
        ctx.set_source_rgb(*fg_color)        
        for xv, yv in centered:
            ctx.move_to(xv[0], yv[0])
            for x, y in zip(xv, yv):
                ctx.line_to(x, y)
            ctx.stroke()

        data = surface.get_data()
        raster_image = np.copy(np.asarray(data)[::4])
        raster_images.append(raster_image.reshape((side, side)) / 255.)
    
    return raster_images

def unpack_drawing(file_handle):
    key_id, = unpack('Q', file_handle.read(8))
    country_code, = unpack('2s', file_handle.read(2))
    recognized, = unpack('b', file_handle.read(1))
    timestamp, = unpack('I', file_handle.read(4))
    n_strokes, = unpack('H', file_handle.read(2))
    image = []
    for i in range(n_strokes):
        n_points, = unpack('H', file_handle.read(2))
        fmt = str(n_points) + 'B'
        x = unpack(fmt, file_handle.read(n_points))
        y = unpack(fmt, file_handle.read(n_points))
        image.append((x, y))
    return image

def unpack_drawings(filename):
    drawings = []
    with open(filename, 'rb') as f:
        while True:
            try:
                drawing = unpack_drawing(f)
                # select 10 random incided in range(len(drawing)), evenly spaced, but must include final 2
                if len(drawing) > 10:
                    indices = np.linspace(0, len(drawing) - 1, 10, dtype=np.int)
                    for i in indices:
                        drawings.append(unpack_drawing(f[:i+1]))
                        drawings.append(drawing)
                else:
                    for i in range(len(drawing)):
                        drawings.append(unpack_drawing(f[:i+1]))
            except struct.error:
                break
    return drawings

def load_train_data(side=256):
    categories = []
    for file in sorted(os.listdir('train')):
        if file.endswith('.ndjson'):
            categories.append(file.split('.')[0])
    OHE = OneHotEncoder(categories=[categories])
    OHE.fit([[category] for category in categories])
            
    train = []
    val = []
    for category in tqdm.tqdm(categories):
        with open(f'train/{category}.ndjson') as f:            
            cat_data = [[torch.tensor(vector_to_raster([scale(shift(rdp_simplify(json.loads(line)["strokes"])))], side=side)[0], dtype=torch.float32), torch.tensor(OHE.transform([[category]]).toarray()[0], dtype=torch.float32)] for line in f]
            random.shuffle(cat_data)
            train.extend(cat_data[:int(len(cat_data) * 0.8)])
            val.extend(cat_data[int(len(cat_data) * 0.8):])
            
    random.shuffle(train)
    random.shuffle(val)
    
    return train, val

class ImageDataset(Dataset):
    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = data             
                        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data_point = self.data[index]
        
        if self.transform:
            dp = (np.expand_dims(data_point[0], axis=0) - 0.12000638813804619) / 0.3186337241490653
            dp = np.repeat(dp, 3, axis=0)
            return [self.transform(dp), data_point[1]]
        else:
            dp = (np.expand_dims(data_point[0], axis=0) - 0.12000638813804619) / 0.3186337241490653
            # repeat three times to get 3 channels
            dp = np.repeat(dp, 3, axis=0)
            return [dp, data_point[1]]
        
'''raw = []
with open('Dataset/tool.ndjson') as f:
    # load from json
    for line in f:
        if len(raw) < 100:
            raw.append(json.loads(line))
        else:
            break
        
processed = [scale(shift(row["strokes"])) for row in raw]
rdp_simplied = [rdp_simplify(row["strokes"]) for row in raw]
rdp_simplied = [scale(shift(row)) for row in rdp_simplied]

imgs_raw = vector_to_raster(processed, side=256, line_diameter=16, padding=16)
imgs_rdp = vector_to_raster(rdp_simplied, side=256, line_diameter=16, padding=16)

for i in range(len(imgs_raw)):
    # plot both images on one plot
    plt.clf()
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(imgs_raw[i].reshape(256, 256))
    axarr[1].imshow(imgs_rdp[i].reshape(256, 256))
    plt.show()
    # save fig
    plt.savefig(f'img/{i}.png')
    plt.close()'''
    
'''
datapoint: {category: str, strokes: list[list[int], list[int]]}
To get picture:
img = vector_to_raster(scale(shift(rdp_simplify(datapoint["strokes"]))), side=256, line_diameter=16, padding=16).reshape(256, 256)

    
strokes = [[[0, 12.781999588012695, 68.16899871826172, 78.59300231933594, 193.21499633789062, 205.7949981689453, 257.64898681640625, 332.3009948730469, 382.6199951171875, 390.12298583984375, 429.135009765625, 500.4070129394531, 574.7620239257812, 598.68798828125, 598.68798828125], [381, 373.0920104980469, 358.75299072265625, 354.5820007324219, 309.35198974609375, 305.1440124511719, 295.81500244140625, 317.7139892578125, 333.4930114746094, 334.9939880371094, 337.2460021972656, 336.4949951171875, 336.4949951171875, 336.4949951171875, 336.4949951171875]], [[0, 0, 0, 3.49399995803833, 32.3129997253418, 46.762001037597656, 55.15299987792969, 188.82899475097656, 275.7919921875, 401.7510070800781, 455.00299072265625, 496.656005859375, 513.1610107421875, 536.4190063476562, 585.9340209960938, 596.43798828125, 596.43798828125], [371, 365, 360, 329.8900146484375, 264.2980041503906, 218.78399658203125, 211.72799682617188, 221.85699462890625, 220.40798950195312, 210.77200317382812, 215.9119873046875, 223.90701293945312, 225.40798950195312, 227.66000366210938, 230.6619873046875, 230.6619873046875, 230.6619873046875]], [[120.78800201416016, 119.96499633789062, 117.78700256347656, 111.0989990234375, 112.53500366210938, 112.53500366210938], [225.40798950195312, 230.34799194335938, 250.177001953125, 322.47198486328125, 331.2409973144531, 331.2409973144531]], [[306.84600830078125, 305.0740051269531, 303.0950012207031, 298.593994140625, 302.3450012207031, 306.09600830078125, 306.09600830078125], [211.89801025390625, 222.53900146484375, 228.7860107421875, 277.1189880371094, 296.7139892578125, 309.4739990234375, 309.4739990234375]], [[465.14599609375, 463.5159912109375, 460.7090148925781, 439.75, 438.9219970703125, 436.6369934082031, 436.6369934082031], [208.14498901367188, 212.22299194335938, 236.47500610351562, 302.3840026855469, 331.58599853515625, 347.7539978027344, 347.7539978027344]]]
#strokes = [[[0, 12.781999588012695, 68.16899871826172, 78.59300231933594, 193.21499633789062, 205.7949981689453, 257.64898681640625, 332.3009948730469, 382.6199951171875, 390.12298583984375, 429.135009765625, 500.4070129394531, 574.7620239257812, 598.68798828125, 598.68798828125], [381, 373.0920104980469, 358.75299072265625, 354.5820007324219, 309.35198974609375, 305.1440124511719, 295.81500244140625, 317.7139892578125, 333.4930114746094, 334.9939880371094, 337.2460021972656, 336.4949951171875, 336.4949951171875, 336.4949951171875, 336.4949951171875]]]
#img = vector_to_raster(scale(shift(rdp_simplify(strokes))), side=256, line_diameter=16, padding=16).reshape(256, 256)
print(strokes)
strokes = rdp_simplify(strokes)
print(strokes)
strokes = scale(shift(strokes))
print(strokes)
img = vector_to_raster([strokes], side=256, line_diameter=16, padding=16)[0].reshape(256, 256)
print(img)'''