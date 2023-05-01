import numpy as np
import pandas as pd
from pathlib import Path
import os.path
import matplotlib.pyplot as plt
import tensorflow as tf

# Create a list with the filepaths for training and testing
train_dir = Path('dataset/train')
train_filepaths = list(train_dir.glob(r'**/*.jpg'))

test_dir = Path('dataset/test')
test_filepaths = list(test_dir.glob(r'**/*.jpg'))

val_dir = Path('dataset/validation')
val_filepaths = list(test_dir.glob(r'**/*.jpg'))



print(test_filepaths[30])

print(str(test_filepaths[30]).replace("\\", "/").split("/")[-2])

# labels = [str(test_filepaths[i]).split("/")[-2] \
#           for i in range(len(test_filepaths))]