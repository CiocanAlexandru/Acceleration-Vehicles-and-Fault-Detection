import numpy as np
import sys 
import keras 
import librosa 
import os 
import pandas as pd 
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import random 
import warnings
from libsvm.svmutil import *
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import make_scorer
from scipy.io.wavfile import read
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.model_selection import KFold
from datetime import datetime
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.datasets import make_classification