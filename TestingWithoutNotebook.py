import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import seaborn as sns
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

with open('data_dump/dataAnalysisVars.pkl', 'rb') as f:
    df = pickle.load(f)

    df_0 = pickle.load(f)
    df_mean = pickle.load(f)
    df_interpolate = pickle.load(f)

    normalized_0 = pickle.load(f)
    scaler_0 = pickle.load(f)

    normalized_mean = pickle.load(f)
    scaler_mean = pickle.load(f)

    normalized_interpolate = pickle.load(f)
    scaler_interpolate = pickle.load(f)