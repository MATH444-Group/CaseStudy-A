import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import time





import __init__

import data_loader as dl
import files





def main():

  data_file = '../data/tests_data.csv'
  output_regression_file = '../images/regression_line.png'
  output_analysis_file = '../docs/results.txt'

  df = dl.load_df(data_file)





  # Build model using the Order of Least Squares algorithm
  model = smf.ols('test2 ~ test1', data=df)
  results = model.fit()

  p_value = results.f_pvalue
  r_squared = results.rsquared
  




  # Plotting data points
  plt.scatter(df['test1'], df['test2'], color='black', label='Actual Data')

  # Draw regression line over the scatter plot we just made
  x_line = np.linspace(df['test1'].min(), df['test1'].max(), 100)
  y_line = results.predict(pd.DataFrame({'test1': x_line}))

  plt.plot(x_line, y_line, color='red', linewidth=2, label='Regression Line')

  plt.title('Test 2 vs Test 1')
  plt.xlabel('Test 1')
  plt.ylabel('Test 2')
  plt.legend()
  plt.grid(True)



  

  # Record our findings in a file for later reference
  files.validate_dir(output_analysis_file)
  with open(output_analysis_file, 'w') as file:
    file.write(results.summary().as_text())
    file.write(f'\n\nFull P-Value: {p_value}')
    file.write(f'\nFull R-Squared: {r_squared}')

  files.validate_dir(output_regression_file)
  plt.savefig(output_regression_file)




if __name__ == '__main__':
  
  logging.info('Began Execution.')
  start = time.time()
  main()
  run_time = time.time() - start
  logging.info(f'Finished Execution.\n\tRun Time: {run_time:.2f}s')