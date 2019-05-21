#%% Imports
import os
try:    
    os.chdir('Assignment_2')
except:
    print("youre already in assignment 2")

import pandas as pd 
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
sns.set_palette("Paired")
from position_bias import *
from pathlib import Path
from average_prop_dest_performance import *

df_train.to_pickle('prepped_df_train.pickle')
df_test.to_pickle('prepped_df_test.pickle')
df_val.to_pickle('prepped_df_val.pickle')