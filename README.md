# a-github-copy-of-FRB-US-Model
A copy of code from https://www.federalreserve.gov/econres/us-models-python.htm

# Step up the enviornment
IMPORTANT: Linux (and Colab) is not supported. Please make sure that you have a Windows OS (tested on Windows 10 19044.1679).
If you are a conda user, you may do the followings:
```
conda create -n frbus python==3.6.0
conda activate frbus
git clone https://github.com/tianyu-z/a-github-copy-of-FRB-US-Model.git
cd a-github-copy-of-FRB-US-Model
python -m pip install -r requirements.txt
```

If you are not a conda user, please make sure you have a Python 3.6.x (tested on 3.6.0). For python version >= 3.7, it will not be functional.  
```
git clone https://github.com/tianyu-z/a-github-copy-of-FRB-US-Model.git
cd a-github-copy-of-FRB-US-Model
pip install -r requirements.txt
```

Please ignore the `setup.py`, it is no longer functional too.

# How to run the demos
```
cd demos
python example1.py
```

Updated on Apr. 27 2022