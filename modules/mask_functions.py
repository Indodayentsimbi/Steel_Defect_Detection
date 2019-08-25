# SOURCE: https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
import pandas as pd
import numpy as np


def mask2rle(mask):
    '''
    mask: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
 
def rle2mask(mask_rle, shape=(1600,256)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    if len(mask_rle) == 0:
        return np.zeros(shape[0]*shape[1], dtype=np.uint8).reshape(shape).T
    else:    
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        mask = np.zeros(shape[0]*shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            mask[lo:hi] = 1
        mask = mask.reshape(shape).T
        return mask



if __name__ == "__main__":

    try:
        assert '1 3 10 5' == mask2rle(rle2mask('1 3 10 5'))
        assert '1 1' == mask2rle(rle2mask('1 1'))
        print('Function mask is good')
    except AssertionError as e:
        print('Error in function mask')