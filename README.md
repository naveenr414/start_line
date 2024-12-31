# Start Line
![Example Plots](img/main_results_2.png)

Starting point for creating projects, comes with plotting, utils, and bash scripts
## Instructions
To start a project, first clone this repoistory
`git clone http://github.com/naveenr414/start-line.git/`
Next, modify `setup.py` to contain the correct project
Similarly, modify the bash script in `scripts/bash_scripts` to contain the correct environment
Then you're all done! 

## Usage
### Plotting
All plotting commands are contained within the `start-line/plotting.py` and `scripts/notebooks/Plotting.ipynb` folders. The notebook provides examples of how to plot common use cases, such as line and bar plots. 

### Running Scripts
Sample scripts are included in the `scripts/bash_scripts` folder. These should save data to the `results/` folder; afterwards, functions in `start-line/utils.py` can help with plotting and processing results. 