# BooleanCryptoGP

The code is based on Python 3.10.11.

Main libraries:
- numpy
- pandas
- matplotlib
- seaborn
- pytexit
- pymoo -> pip3 install -U pymoo
- genepro 1.3.1 -> https://github.com/giorgia-nadizar/genepro/ (download this repo locally, put yourself within the folder and type "pip3 install -U ." to install this library, you must be in the folder that contains the setup.py file to do so)

Scripts to launch experiments are available in the exps package.
Mind that you should set up an environment variable in your system called CURRENT_CODEBASE_FOLDER in which you store the absolute path of the folder containing python code and python data, including this repository (this path should end with '/'). Check if the path typed in the experiment script is correct for you. Also check that the PYTHONPATH variable is correctly set up.

Results are saved as .csv file. Each experiment generates three csv files containing the following data: optimal pareto front, population statistics over the evolution, pareto front for each generation.

