# ImageSeg_LeafandSoil
Refactoring and parallelization of code for Devin Rippner, USDA, to train image segmentation models of leaf and soil tomographic data.


## Instructions

### Preparing the environment

I would suggest using `conda` to make the correct python environment. 

If `conda` is installed (is available on NERSC systems) simply use the following: 

`conda env create -n {ENVNAME} --file environment.yml` with a custom `ENVNAME`.

### Inputs to model training

The classes are configured to take an input dictionary. The input dictionary is available to view and modify in both `.ipynb` examples, and the `train.py` script. Please make changes to the directories to point to the appropriate data, and update the number of epochs to train. 

### Running on Perlmutter. 

The notebooks can be run with the appropriate conda environment with the following set up: 
[Step Up Instructions from NERSC Docs](https://docs.nersc.gov/development/languages/python/faq-troubleshooting/#can-i-use-my-conda-environment-in-jupyter)

Request a notebook on an exclusive node to get 4 GPUS for parallel training. 

To submit as a batch job for scheduled computation, prepare the inputs in `train.py` as normal. 

In a terminal window, execute: 

`sbatch run.sh`

After submission, you can check the status of the job using: 

`squeue -u {USERNAME}`

Please contact [Lipi Gupta](lipigupta@lbl.gov) with questions. 
