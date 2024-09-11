"""
Exposes common paths useful for manipulating datasets and generating figures.

"""
from pathlib import Path
import matplotlib
from matplotlib import font_manager


LBL_FONTSIZE = 12.5



# Absolute path to the top level of the repository
src = Path(__file__).resolve().parents[1].absolute()

data = src / "data"
scripts = src / "scripts"
figures = src / "figures"
rc_file = scripts / "matplotlibrc"
font_file = scripts / "LiberationSans.ttf"

font_file = Path(font_file)
if font_file.exists():
    font_manager.fontManager.addfont(str(font_file))
else:
    print(f"Font file not found: {font_file}")

matplotlib.rc_file(rc_file)