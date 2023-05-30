import numpy as np
from cv2 import resize, imread



def xrgb2gray(img):
    assert len(img.shape) in (2, 3)
    return img.mean(axis=2) if len(img.shape) == 3 else img
        

def im2float(img, dtype=np.float64):
    if np.issubdtype(img.dtype, np.integer):
        return img.astype(dtype) / np.iinfo(img.dtype).max
    return img


# Returns SM in [0; 1] range
def read_sm(path):
    return xrgb2gray(im2float(imread(path, 0)))


def padding(img, height=360, width=640, channels=3):
    channels = img.shape[2] if len(img.shape) > 2 else 1

    if channels == 1:
        img_padded = np.zeros((height, width), dtype=img.dtype)
    else:
        img_padded = np.zeros((height, width, channels), dtype=img.dtype)

    original_shape = img.shape
    rows_rate = original_shape[0] / height
    cols_rate = original_shape[1] / width

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * height) // original_shape[0]
        img = resize(img, (new_cols, height))
        if new_cols > width:
            new_cols = width
        img_padded[:, ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols)] = img

    else:
        new_rows = (original_shape[0] * width) // original_shape[1]
        img = resize(img, (width, new_rows))
        if new_rows > height:
            new_rows = height
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded


def resize_fixation(img, rows=216, cols=384):
    out = np.zeros((rows, cols))
    factor_scale_r = rows / img.shape[0]
    factor_scale_c = cols / img.shape[1]

    coords = np.argwhere(img)

    for coord in coords:
        r = int(np.round(coord[0] * factor_scale_r))
        c = int(np.round(coord[1] * factor_scale_c))
        out[min(r, rows-1), min(c, cols-1)] = 255

    return out.astype(int)


def padding_fixation(img, height=360, width=640):
    img_padded = np.zeros((height, width))

    original_shape = img.shape
    rows_rate = original_shape[0] / height
    cols_rate = original_shape[1] / width

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * height) // original_shape[0]
        img = resize_fixation(img, rows=height, cols=new_cols)
        if new_cols > width:
            new_cols = width
        img_padded[:, ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols)] = img

    else:
        new_rows = (original_shape[0] * width) // original_shape[1]
        img = resize_fixation(img, rows=new_rows, cols=width)
        if new_rows > height:
            new_rows = height

        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded
