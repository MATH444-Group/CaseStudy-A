import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import statsmodels.formula.api as smf
import sys
import time




relative_git_log_file = '../logs/git.log'
relative_git_log_dir = '../logs'

if not os.path.isdir(relative_git_log_dir):
  os.mkdir(relative_git_log_dir)

if not os.path.isdir('utils'):
  os.system(f'printf \'{time.asctime()}:\\n\' > {relative_git_log_file} 2>&1')
  os.system(f'git clone https://github.com/GevChalikyan/utils.git >> {relative_git_log_file} 2>&1')

else:
  os.chdir('utils')
  relative_git_log_file = '../' + relative_git_log_file
  
  os.system(f'printf \'{time.asctime()}:\\n\' >> {relative_git_log_file} 2>&1')
  os.system(f'git pull >> {relative_git_log_file} 2>&1')
  os.chdir('..')

sys.path.append('utils')
import data_loader as dl





def main():

  data_file_path = '../data/tests_data.csv'
  output_regression_file_path = '../images/regression_line.png'
  output_regression_dir_path = '../images'

  df = dl.load_df(data_file_path)

  model = smf.ols('test2 ~ test1', data=df)
  results = model.fit()
  
  plt.scatter(df['test1'], df['test2'], color='black', label='Actual Data')

  x_line = np.linspace(df['test1'].min(), df['test1'].max(), 100)
  y_line = results.predict(pd.DataFrame({'test1': x_line}))

  plt.plot(x_line, y_line, color='red', linewidth=2, label='Regression Line')

  plt.title('Test 2 vs Test 1')
  plt.xlabel('Test 1')
  plt.ylabel('Test 2')
  plt.legend()
  plt.grid(True)

  print(results.summary())

  if not os.path.isdir(output_regression_dir_path):
    os.mkdir(output_regression_dir_path)
  plt.savefig(output_regression_file_path)




if __name__ == '__main__':

  import logger
  logger.setup()

  main()