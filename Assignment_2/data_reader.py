#%%
import pandas as pd 

# Reading in db
df = pd.read_csv('Assignment_2/data/training_set_VU_DM.csv')

#%%
smalldf = df.head(10000)


#%%
df.prop_id.value_counts()

#%%
