import numpy as np
from aicsimageio import AICSImage
import skimage.filters as skif

def ReadFunction(path_to_file: str) -> np.arrays:
    """Read in file and perform following actions: create mean projection, subtract mean projection from original movie.
        Create min projection of processed movie and calculate otsu threshold. Crate binary movie with threshold*0.8

    Args: 
        path to file

    Return:
        Original Movie
        Binary Movie
    """
    #read in one image stack
    img_movie = AICSImage(path_to_file)
    #create the mean projection of the movie
    img_mean = np.mean(img_movie.get_image_data("TYX"), axis = 0)
    #subtract mean projection of the movie
    movie_mean = img_movie.data - img_mean
    #create min projection of the previous processed movie
    img_mean_min = np.min(movie_mean, axis = 0)
    #calculate the threshold of the min projeciton with otsu
    threshold = skif.threshold_otsu(img_mean_min)
    #create binary movie with threshold
    movie_binary = movie_mean < (threshold*0.8)

    return img_movie, movie_binary