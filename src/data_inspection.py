from pathlib import Path
import pandas as pd

#* set data path
dataRoot = Path.cwd().parent/'data'

#* read in data
df = pd.read_csv(dataRoot/'data.csv')

#* Facts 
#how many missing values per column
missings = df.isnull().sum()
