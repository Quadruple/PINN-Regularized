# PINN-Regularized
This repository belongs to an experiment that applies L1 and L2 regularizations to a Physics Informed Neural Network. The code records the L1, L2 and L-inf
losses for 3 cases as L1 regularized, L2 regularized and non-regularized. Then, by processing the outputs of the scripts several scatter plots are plotted
for comparing the regularized versions of the neural network to the non-regularized one.

In order to run the scripts and process the results related to losses please follow the below instructions:

1. Create a conda environment based on the requirements.txt file. For environment creation please follow the comment on top of requirements.txt file.
2. Activate the conda environment:   
  `conda activate <your-env-name>`
3. Go into each directory (l1, l2, non_regularized) and run the script that starts with    
`python HJB-PINN-<loss-type>.py`
4. The scripts are designed such that they will output to the console with print statements. If you would like to process the data, please 
redirect script outputs to a file as:    
`python -u HJB-PINN-<loss-type>.py > pinn_run_output.txt`
5. After recording the outputs into corresponding folders and .txt files, run the data_process.py.    
`python data_processor.py` 
6. All the tracked losses with and without regularization will be scatter plotted into their corresponding figure directory. For figures, check the folder
"figures".

**NOTE**: If you would like to process different output files, change the paths defined inside `data_processor.py`
