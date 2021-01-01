# <center>NBA GAME ATTENDANCE ANALYSIS</center>

### Usage 
if anaconda is already installed (conda commands can already be run):
- if local conda environment not yet created (to check `terminal:` conda env list ; ieor_142_project should not appear in list):
	- navigate to directory of repo in terminal
    	- cd to change directory, ls to list files in directory
    	- or cd then drag folder into terminal (this will copy folder path)
  	- `terminal:` conda env create -f environment.yml 
   		- This should lead to packages being installed in the new conda environment
  	- ensure that environment is activated ( (ieor_142_project) should appear at beginning of terminal prompt)
   		- if not, then: `terminal:` conda activate ieor_142_project
- else if local conda environemnt is already created (to check `terminal:` conda env list ; ieor_142_project should appear in list):
	- ensure that environment is activated ( (ieor_142_project) should appear at beginning of terminal prompt)
   		- if not, then: `terminal:` conda activate ieor_142_project
	- `terminal:` conda env update --prefix ./env --file environment.yml  --prune
		- This should lead to any missing packages being installed
		
else if anaconda not yet installed (conda commands cannot be run):
- follow instructions here: https://docs.conda.io/en/latest/miniconda.html
- Once installed follow instructions above
 
