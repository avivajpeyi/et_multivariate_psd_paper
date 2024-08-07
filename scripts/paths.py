"""
Exposes common paths useful for manipulating datasets and generating figures.

"""
from pathlib import Path
import matplotlib
from matplotlib import font_manager


# Absolute path to the top level of the repository
src = Path(__file__).resolve().parents[1].absolute()

data = src / "data"
scripts = src / "scripts"
figures = src / "figures"
rc_file = scripts / "matplotlibrc"
font_file = scripts / "LiberationSans.ttf"

font_manager.fontManager.addfont(font_file)
matplotlib.rc_file(rc_file)
