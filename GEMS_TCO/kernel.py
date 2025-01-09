import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal # for simulation
# !pip install scipy
import math 
import scipy
from scipy.special import kv # for 
from collections import defaultdict
from scipy.spatial.distance import cdist # for space and time distance

from scipy.spatial import distance # find closest spatial point

import sklearn.neighbors  # nearest neighbor
from typing import Callable   # nearest neighbor function input type

import random # randomy select  random.sample
from skgstat import Variogram