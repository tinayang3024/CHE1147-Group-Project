import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#read parquet dataset
enzy_data = pd.read_parquet(r"/content/TheData_kcat.parquet")

#descibe data info
print(enzy_data.describe)
print(enzy_data.describe)

