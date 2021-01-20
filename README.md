# Machine Learning for NBA Game Attendance Prediction

This project seeks to provide a tool to accurately predict the attendance of NBA games in order to better inform the business decisions of different stakeholders across the organization. Predicting game attendance is crucial to making optimized managerial decisions such as planning necessary staffing needs or procuring the proper level of supplies (janitorial, food services, etc). The project is currently being worked on in its second version, `version_2`. In version 1, an entire machine learning pipeline is established throughout a host of modules ranging from web scraping for data collection to neural-network regression modeling for prediction. These efforts resulted in a high accuracy model with mean absolute error values for attendance around 800 people. However, improvements in data sources and modeling paradigms for improved accuracy are being sought in a few ways in the upcoming version. Click the link below to view the analysis and modeling version 1.0 notebook or continue reading for more about the project. 

<p align="center">
  Interact with the project notebook in your web browser using the <i>Binder</i> service  
<a target="_blank" rel="noopener noreferrer" href=https://mybinder.org/v2/gh/wyattowalsh/NBA-attendance-prediction/HEAD?filepath=nb.ipynb> <img src=https://mybinder.org/badge_logo.svg></a>
 <br><br>
</p>

![](notebook_preview.gif)

---

## Contents

- [Explanantion of Repository Contents](#Explanantion-of-Repository-Contents)
- [Version 2.0](#Version-2.0)
    - [Development Roadmap](#Development-Roadmap)
    - [Progress Updates](#Progress-Updates)
- [Version 1.0](#Version-1.0)
    - [Project Summary](#Project-Summary)
    - [Results and Discussion](#Results-And-Discussion)
- [Installation Instructions](#Installation-Instructions)

---

## Explanation of Repository Contents

- `data` contains both raw and processed data. There are game, search popularity, and stadium wiki raw datasets. These three datasets are processed and compiled resulting in the file `dataset.csv` within the `processed` directory. However, numerous other datasets can be found here which are the accumulation of different feature selection and data sampling strategies for use in modeling. 
- `features` contains results derived from statistical testing and principal components analysis across the datasets
- `models` contains datasets of the error results across all the models applied as well as tuning parameter values
- `src` is where all the project source code can be found. A host of modules and functions for data web scraping, feature selection, visualization, modeling, and Jupyter configuration are here. 
- `version_2` is where all files related to the second iteration of this project can be found. Its structure generally mirrors that of repository root directory with sub-directories for data, source code, etc. 
- `visualizations` holds .png images of the visualizations created on the datasets
- `nb.ipynb` is the associated data analysis and modeling notebook (this notebook can also be found and interacted with via the Binder link found above.
- `r_modeling.ipynb` is an R notebook used for further data modeling with more exotic models. 
- `environment.yml` and `requirements.txt` are environment setup files to properly configure an environment and load necessary dependendicies for the project (a further explanation of how to use these can be found at the bottom)

--- 

## Version 2.0

### Development Roadmap

The goal of this version is to create another implementation of this machine learning pipeline leveraging knowledge gained from the first version to improve overall predictive accuracy and utilize new tools and modeling techniques. 

To avoid any potential data cleanliness problems with scraping data from [basketball-reference.com](https://www.basketball-reference.com/) as in the first version, [stats.nba.com](https://www.nba.com/stats/) will be queried through an open source API for sport related data enabling more seasons and features to be gathered. Furthermore, a wider range of data sources will be considered taking into consideration factors such as regional socioeconomics, weather, etc. New pre-processing scripts will be used to combine and clean the data from these different sources in order to make a dataset apt for modeling. Core modeling assumptions leveraged in the first version, such as data distribution will be re-evaluated. Furthermore, a new portfolio of modeling techniques based on more current research will be applied. A few models to be included are linear regression with the Huber loss, a long short-term memory neural network, and ensemble methodologies.

In future versions, a full Kubernetes cluster of the pipeline deployed via distributed cloud-computing resources would be a wonderful addition. This would allow for automated model updates, fully parallelized modeling (as every model can be containerized), and prediction delivery. 

### Progress Updates

- Game data and especially attendance data was successfully retrieved for all seasons since 1946 using [nba_api](https://github.com/swar/nba_api). This is awesome as version 1.0 only included seasons since 1999. The package was discovered on Github and leveraged to query the numerous [stats.nba.com]() endpoints. 
- Functions to query different types of datasets as well as functions to combine and clean the results have been created for league team data, game overiew data for all seasons, and game box score summary data for all games. 

--- 

## Version 1.0

### Project Summary

As briefly discussed in the introduction, the project's aim is to create an NBA game attendance prediction tool in order to improve the business decisions of NBA stadium managers. These managers have to make dynamic decisions reacting to fluctuating demand in a constrained, complex environment. Staff scheduling, food services, and entertainment are just a few of these decision areas. Game attendance predictions can be used as a tool to gain insights on customer demand and help to better inform these manager's decisions. Operating expenses can be minimized if waste is minimized and properly assessing demand helps to ensure fewer overages. Assessing demand can further impact the bottom line of the stadium through helping to ensure there are proper supply levels to meet customer demand. 

Game attendance prediction can serve to underlie many of the tools and processes found across the different facets of the organization. As an example, vendors can use attendance predictions along with their own demand metrics and analytics, to better assess how many soft goods to purchase. Simililar foundational relationships can be found for most vendors, as well as janitorial supplies, process timing, and facility operations. 

The flowchart below details the five different stages within the pipeline architecture used here.

![alt text](https://mermaid.ink/svg/eyJjb2RlIjoiZ3JhcGggTFJcbiAgICBBW0EuIERhdGEgSW5nZXN0aW9uXSAtLT4gQltCLiBEYXRhIFByZS1wcm9jZXNzaW5nXVxuICAgIEIgLS0-IENbQy4gRXhwbG9yYXRvcnkgRGF0YSBBbmFseXNpc11cbiAgICBDIC0tPiBEW0QuIE1vZGVsaW5nXVxuICAgIEQgLS0-IEVbRS4gRXZhbHVhdGlvbl0iLCJtZXJtYWlkIjp7InRoZW1lIjoiZm9yZXN0IiwidGhlbWVWYXJpYWJsZXMiOnsiYmFja2dyb3VuZCI6IndoaXRlIiwicHJpbWFyeUNvbG9yIjoiIzkxQjNFRiIsInNlY29uZGFyeUNvbG9yIjoiI2ZmZmZkZSIsInRlcnRpYXJ5Q29sb3IiOiJoc2woODAsIDEwMCUsIDk2LjI3NDUwOTgwMzklKSIsInByaW1hcnlCb3JkZXJDb2xvciI6ImhzbCgyNDAsIDYwJSwgODYuMjc0NTA5ODAzOSUpIiwic2Vjb25kYXJ5Qm9yZGVyQ29sb3IiOiJoc2woNjAsIDYwJSwgODMuNTI5NDExNzY0NyUpIiwidGVydGlhcnlCb3JkZXJDb2xvciI6ImhzbCg4MCwgNjAlLCA4Ni4yNzQ1MDk4MDM5JSkiLCJwcmltYXJ5VGV4dENvbG9yIjoiIzEzMTMwMCIsInNlY29uZGFyeVRleHRDb2xvciI6IiMwMDAwMjEiLCJ0ZXJ0aWFyeVRleHRDb2xvciI6InJnYig5LjUwMDAwMDAwMDEsIDkuNTAwMDAwMDAwMSwgOS41MDAwMDAwMDAxKSIsImxpbmVDb2xvciI6IiMzMzMzMzMiLCJ0ZXh0Q29sb3IiOiIjMzMzIiwibWFpbkJrZyI6IiNFQ0VDRkYiLCJzZWNvbmRCa2ciOiIjZmZmZmRlIiwiYm9yZGVyMSI6IiM5MzcwREIiLCJib3JkZXIyIjoiI2FhYWEzMyIsImFycm93aGVhZENvbG9yIjoiIzMzMzMzMyIsImZvbnRGYW1pbHkiOiJcInRyZWJ1Y2hldCBtc1wiLCB2ZXJkYW5hLCBhcmlhbCIsImZvbnRTaXplIjoiMTZweCIsImxhYmVsQmFja2dyb3VuZCI6IiNlOGU4ZTgiLCJub2RlQmtnIjoiI0VDRUNGRiIsIm5vZGVCb3JkZXIiOiIjOTM3MERCIiwiY2x1c3RlckJrZyI6IiNmZmZmZGUiLCJjbHVzdGVyQm9yZGVyIjoiI2FhYWEzMyIsImRlZmF1bHRMaW5rQ29sb3IiOiIjMzMzMzMzIiwidGl0bGVDb2xvciI6IiMzMzMiLCJlZGdlTGFiZWxCYWNrZ3JvdW5kIjoiI2U4ZThlOCIsImFjdG9yQm9yZGVyIjoiaHNsKDI1OS42MjYxNjgyMjQzLCA1OS43NzY1MzYzMTI4JSwgODcuOTAxOTYwNzg0MyUpIiwiYWN0b3JCa2ciOiIjRUNFQ0ZGIiwiYWN0b3JUZXh0Q29sb3IiOiJibGFjayIsImFjdG9yTGluZUNvbG9yIjoiZ3JleSIsInNpZ25hbENvbG9yIjoiIzMzMyIsInNpZ25hbFRleHRDb2xvciI6IiMzMzMiLCJsYWJlbEJveEJrZ0NvbG9yIjoiI0VDRUNGRiIsImxhYmVsQm94Qm9yZGVyQ29sb3IiOiJoc2woMjU5LjYyNjE2ODIyNDMsIDU5Ljc3NjUzNjMxMjglLCA4Ny45MDE5NjA3ODQzJSkiLCJsYWJlbFRleHRDb2xvciI6ImJsYWNrIiwibG9vcFRleHRDb2xvciI6ImJsYWNrIiwibm90ZUJvcmRlckNvbG9yIjoiI2FhYWEzMyIsIm5vdGVCa2dDb2xvciI6IiNmZmY1YWQiLCJub3RlVGV4dENvbG9yIjoiYmxhY2siLCJhY3RpdmF0aW9uQm9yZGVyQ29sb3IiOiIjNjY2IiwiYWN0aXZhdGlvbkJrZ0NvbG9yIjoiI2Y0ZjRmNCIsInNlcXVlbmNlTnVtYmVyQ29sb3IiOiJ3aGl0ZSIsInNlY3Rpb25Ca2dDb2xvciI6InJnYmEoMTAyLCAxMDIsIDI1NSwgMC40OSkiLCJhbHRTZWN0aW9uQmtnQ29sb3IiOiJ3aGl0ZSIsInNlY3Rpb25Ca2dDb2xvcjIiOiIjZmZmNDAwIiwidGFza0JvcmRlckNvbG9yIjoiIzUzNGZiYyIsInRhc2tCa2dDb2xvciI6IiM4YTkwZGQiLCJ0YXNrVGV4dExpZ2h0Q29sb3IiOiJ3aGl0ZSIsInRhc2tUZXh0Q29sb3IiOiJ3aGl0ZSIsInRhc2tUZXh0RGFya0NvbG9yIjoiYmxhY2siLCJ0YXNrVGV4dE91dHNpZGVDb2xvciI6ImJsYWNrIiwidGFza1RleHRDbGlja2FibGVDb2xvciI6IiMwMDMxNjMiLCJhY3RpdmVUYXNrQm9yZGVyQ29sb3IiOiIjNTM0ZmJjIiwiYWN0aXZlVGFza0JrZ0NvbG9yIjoiI2JmYzdmZiIsImdyaWRDb2xvciI6ImxpZ2h0Z3JleSIsImRvbmVUYXNrQmtnQ29sb3IiOiJsaWdodGdyZXkiLCJkb25lVGFza0JvcmRlckNvbG9yIjoiZ3JleSIsImNyaXRCb3JkZXJDb2xvciI6IiNmZjg4ODgiLCJjcml0QmtnQ29sb3IiOiJyZWQiLCJ0b2RheUxpbmVDb2xvciI6InJlZCIsImxhYmVsQ29sb3IiOiJibGFjayIsImVycm9yQmtnQ29sb3IiOiIjNTUyMjIyIiwiZXJyb3JUZXh0Q29sb3IiOiIjNTUyMjIyIiwiY2xhc3NUZXh0IjoiIzEzMTMwMCIsImZpbGxUeXBlMCI6IiNFQ0VDRkYiLCJmaWxsVHlwZTEiOiIjZmZmZmRlIiwiZmlsbFR5cGUyIjoiaHNsKDMwNCwgMTAwJSwgOTYuMjc0NTA5ODAzOSUpIiwiZmlsbFR5cGUzIjoiaHNsKDEyNCwgMTAwJSwgOTMuNTI5NDExNzY0NyUpIiwiZmlsbFR5cGU0IjoiaHNsKDE3NiwgMTAwJSwgOTYuMjc0NTA5ODAzOSUpIiwiZmlsbFR5cGU1IjoiaHNsKC00LCAxMDAlLCA5My41Mjk0MTE3NjQ3JSkiLCJmaWxsVHlwZTYiOiJoc2woOCwgMTAwJSwgOTYuMjc0NTA5ODAzOSUpIiwiZmlsbFR5cGU3IjoiaHNsKDE4OCwgMTAwJSwgOTMuNTI5NDExNzY0NyUpIn19LCJ1cGRhdGVFZGl0b3IiOmZhbHNlfQ)

- A: Stadium data (e.g. location) is scraped from [wikipedia](https://www.wikipedia.org/) using the Pandas library. Game and sport data is scraped from [basketball-reference.com](https://www.basketball-reference.com/) using the Beautiful Soup framework and requests library. The [pytrends](https://github.com/GeneralMills/pytrends) Google Trends API is used to gather search popularity data. 
- B: 


### Results and Discussion

## Installation Instructions

`environment.yml`  can be found in the repository's root directory for your version of interest and used to install necessary project dependencies. If able to successfully configure your computing environment, then launch Jupyter Notebook from your command prompt and navigate to `nb.ipynb`. If unable to successfully configure your computing environment refer to the sections below to install necessary system tools and package dependencies. The following sections may be cross-platform compatibile in several places, however is geared towards macOS<sup>[1](#footnote1)</sup>.

#### Do you have the Conda system installed?

Open a command prompt (i.e. *Terminal*) and run: `conda info`.

This should display related information pertaining to your system's installation of Conda. If this is the case, you should be able to skip to the section regarding virtual environment creation (updating to the latest version of Conda could prove helpful though: `conda update conda`).

If this resulted in an error, then install Conda with the following section. 

#### Install Conda

There are a few options here. To do a general full installation check out the [Anaconda Download Page](https://docs.conda.io/projects/conda/en/latest/user-guide/install/). However, the author strongly recommends the use of Miniconda since it retains necessary functionality while keeping resource use low; [Comparison with Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html#anaconda-or-miniconda) and [Miniconda Download Page](https://docs.conda.io/en/latest/miniconda.html). 

Windows users: please refer to the above links to install some variation of Conda. Once installed, proceed to the instructions for creating and configuring virtual environments [found here](#Configure-Local-Environment

macOS or Linux users: It is recommended to use the [Homebrew system](https://brew.sh/) to simplify the Miniconda installation process. Usage of Homebrew is explanained next. 

##### Do you have Homebrew Installed?

In your command prompt (i.e. *Terminal*) use a statement such as: `brew help`

If this errored, move on to the next section.

If this returned output (e.g. examples of usage) then you have Homebrew installed and can proceed to install conda [found here](#Install-Miniconda-with-Homebrew).

##### Install Homebrew

In your command prompt, call: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`

##### Install Miniconda with Homebrew

In your command prompt, call: `brew install --cask miniconda`

When in doubt, calling in the `brew doctor` might help :pill: 

##### A Few Possibly Useful Conda Commands

All environment related commands can be found here: [Anaconda Environment Commands](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

Here are a few of the most used ones though: 

List all environments (current environment as marked by the \*): `conda env list`

Create a new environment: `conda create --name myenv`

Activate an environment: `conda activate myenv`

Deactivate an environment and go back to system base: `conda deactivate`

List all installed packages for current environment: `conda list`

#### Configure Local Environment

Using the command prompt, navigate to the local project repository directory -- On macOS, I recommend typing `cd ` in Terminal and then dragging the project folder from finder into Terminal. 

In your command prompt, call: `conda env create -f environment.yml`. This will create a new Conda virtual environment with the name: `NBA-attendance-prediction`.

Activate the new environment by using: `conda activate NBA-attendance-prediction`

#### Access Project

After having activated your environment, use `jupyter notebook` to launch a Jupyter session in your browser. 

Within the Jupyter Home page, navigate and click on `nb.ipynb` in the list of files. This will launch a local kernel running the project notebook in a new tab. 

---
<br></br>
<br></br>
<br></br>
<br></br>
<br></br>
<br></br>
<br></br>

<a name="footnote1">1</a>: This project was created on macOS version 11.0.1 (Big Sur) using Conda version 4.9.2, and Python 3.8 (please reach out to me if you need further system specifications). 
