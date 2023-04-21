# dag_prf_utils repository

This repository is a collection of useful tools for analysing and visualising PRF data. It is designed to work with prfpy https://github.com/VU-Cog-Sci/prfpy, but with some of the tools are general purpose. 

## A mini overview 


### utils
Mainly generic tools - for searching for files, some mathematical stuff which comes up again and again in PRF analyses (converting between cartesian and polar coordinates etc; calculating rsq...)
In addition there are a couple of *prfpy specific tools*, i.e., Prf1T1M, etc., which are designed to load the outputs of prfpy models and hold them in easily manipulatable ways.

### plot functions
Several functions useful for plotting PRF properties around the visual field. All focus on matplotlib

### surface plotting and meshes



## In active development
Stuff may well change. 

## Installation
Standard git install 
Then setup using `python setup.py develop`
