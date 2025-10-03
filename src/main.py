import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf





import __init__

import data_loader as dl





def main():

  data_file_path = '../data/tests_data.csv'
  output_regression_file_path = '../images/regression_line.png'
  output_regression_dir_path = '../images'
  output_analysis_file_path = '../docs/results.txt'
  output_analysis_dir_path = '../docs'



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

  p_value = results.f_pvalue
  r_squared = results.rsquared

  if not os.path.isdir(output_analysis_dir_path):
    os.mkdir(output_analysis_dir_path)
  with open(output_analysis_file_path, 'w') as file:
    file.write(results.summary().as_text())
    file.write(f'\n\nFull P-Value: {p_value}')
    file.write(f'\nFull R-Squared: {r_squared}')

  if not os.path.isdir(output_regression_dir_path):
    os.mkdir(output_regression_dir_path)
  plt.savefig(output_regression_file_path)




if __name__ == '__main__':

  import logger
  logger.setup()

  main()