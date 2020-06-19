# LTO-CMA
Code for the paper "Learning Step-Size Adaptation in CMA-ES"
## Experiment Setup
### Training
- Create experiment folder
- Create file with hyperparameters of the experiment *hyperparams.py* in the experiment folder
- Start learning step-size adaptation by executing the command:
```
python gps_main.py EXPERIMENT_FOLDER_NAME
```
- The output of training is the pickled version of the learned policy, saved in the path *EXPERIMENT_FOLDER_NAME/data_files*.
### Testing
- Add the path to the learned policy in the hyperparameter file *hyperparams.py*
-Start testing the performance of the learned policy on the test set by executing the command:
```
python gps_test.py EXPERIMENT_FOLDER_NAME
```
- The output of testing are the files *test_data_X.json* for each condition index X of the test set, saved in the experiment folder.
- The output file *test_data_X.json* contains:
  - The average objective values from 25 samples of running the learned policy on the test condition X,
  - The end objective values of the 25 samples,
  - The average step-size for each step of the optimization trajectory from 25 samples, and 
  - The standard deviation of the objective value and the step-size for each step of the optimization trajectory.
- To plot the results, run the *plot_performance.py* script in the *scripts* folder.
