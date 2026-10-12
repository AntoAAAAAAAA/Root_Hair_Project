import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
import scipy as scipy
import plotly as plot
import plotly.express as px
from ultralytics import SAM

from src.main_root_mask import *
from src.root_hairs_mask import *
from src.final_visualize import *
from src.ML import ML_model

def main_v3(color_image, image_gray, microscope_conversion_factor, upper, lower, model):

    fig, traces = ML_model(image_gray, model)

    ...