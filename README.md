### Lab For Data Mining 2017 Fall @ NTHU
This repository contains all the instructions and necessary code for Data Mining 2017 (Fall) lab session.

---

### Computing Resources
- Operating system: Preferably Linux or MacOS
- RAM: 8GB
- Disk space: Minimum 8GB

---
### Software Requirements
Here is a list of the required programs and libraries necessary for this lab session:
- [Python 3+](https://www.python.org/download/releases/3.0/) (Note: coding will be done strictly on Python 3)
    - Install latest version of Python 3
- [Anaconda](https://www.anaconda.com/download/) environemnt or any other environement (recommended but not required)
    - Install anaconda environment
- [Jupyter](http://jupyter.org/) (Strongly recommended but not required)
    - Install jupyter
- [Scikit Learn](http://scikit-learn.org/stable/index.html)
    - Install `sklearn` latest python library
- [Pandas](http://pandas.pydata.org/)
    - Install `pandas` python library
- [Numpy](http://www.numpy.org/)
    - Install `numpy` python library
- [Matplotlib](https://matplotlib.org/)
    - Install `maplotlib` for python
- [Plotly](https://plot.ly/)
    - Install and signup for `plotly`
- [NLTK](http://www.nltk.org/)
    - Install `nltk` library
- [WordCloud](https://github.com/amueller/word_cloud)
    - Install library for generating word clouds

---
### Test script
Open a Jupyter notebook and run the following commands. If you have properly installed all the necessary libraries you should see no error.
```python
import pandas as pd
import numpy as np
import nltk
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import plotly.plotly as py
import plotly.graph_objs as go
import math
%matplotlib inline
from wordcloud import WordCloud
# my functions
import helpers.data_mining_helpers as dmh
import helpers.text_analysis as ta
```

---
### Preview of Complete Jupyter Notebook
https://github.com/omarsar/data_mining_2017_fall_lab/blob/master/news_data_mining.ipynb
