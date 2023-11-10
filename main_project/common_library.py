import numpy as np
import sys 
import keras 
import librosa 
import os 
import pandas as pd 
from scipy.io.wavfile import read
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from datetime import datetime
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.datasets import make_classification