# dag_prf_utils repository

This repository is a collection of useful tools for analysing and visualising PRF data. It is designed to work with prfpy https://github.com/VU-Cog-Sci/prfpy, but with some of the tools are general purpose. 

## A mini overview 


### utils
Mainly generic tools - for searching for files, some mathematical stuff which comes up again and again in PRF analyses (converting between cartesian and polar coordinates etc; calculating rsq...).

In addition there are a couple of *prfpy specific tools*, i.e., Prf1T1M, etc., which are designed to load the outputs of prfpy models and hold them in easily manipulatable ways.

### plot functions
Several functions useful for plotting PRF properties around the visual field. All focus on matplotlib

### surface plotting and meshes
I have created several tools for plotting data on the cortical surface. This will of course require that you have surface data (i.e., run freesurfer); and any data you want to plot will need to be in a vertex wise format. 
pycortex (https://github.com/gallantlab/pycortex) is a very powerful way of plotting information on the cortical surface and an excellent tool. 

...but sometimes you just want to do simple stuff, and customize certain properties aspects (e.g., color maps etc.). Also it can be useful to have the cortical mesh in a generic format readable by any 3D rendering software (e.g., .ply), and in a way that you can send a single file anywhere, and it be instantly readable, with color and everything. 

To this end I have created a couple of different ways to visualize surface information. 

[1] Freeview based surface plotting
This requires freeview, and nibabel. The script will take custom vertex-wise data (e.g., eccentricity estimates), combine it with a mask (to hide the bad prf fits). Then create a custom surf file, which you can load in freeview. 
You can also specify any matplotlib color map, with any range of values to show this. The script will produce a command to load this in freeview. 
You can also specify the camera angle for when freeview opens, and ask it to automatically take a picture of the surface. This can be useful if you want to iterate through several subjects/surface plots and save the figures as pngs, but can't be bothered to sit and click again and again... 

Note one limitation is there is no option for alpha masking (i.e., varying the levels of transparency of the data on the cortex; it is just binary)

[2] Generic .ply format
If you want to have 

Experimental way to view MRI surface data (without pycortex; e.g., to view retinotopic maps)
> why do this? 
Pycortex is very powerful, but also quite complex. The source code is difficult to follow, and when it doesn't work; it is difficult to find out why. The idea here is to have a simple script which allows you to plot data on the cortical surface quickly. It should also allow you to specify you're own custom color maps. It (hopefully) allows you to view you're surface in a 3D software package of your choice. Here I am using meshlab. 

You can install Meshlab, and specify the path to run the function. 

What it does: 
[1] For a subject, take a freesurfer surface (e.g., pial, or inflated), convert it into a "mesh file" format which can be easily read by standard 3D rendering software (e.g.,".ply", using meshlab)

[2] Render some anatomical properties of this data on the surface (e.g., the curvature, or sulcal depth) 

[3] Plot arbitrary data on the surface (e.g., retinotopic stuff, like polar angle)
> this can be any values (of a length which matches the number of vertices), specified by the user
> you can specify any matplotlib colormap
> and you can specify the alpha, allowing it to nicely blend with the anatomical data (e.g. the curvature)

This is all saved in a .ply file, and can be viewed using meshlab
> you can also click on individual vertices inside meshlab, to get there position, index, and values (of the data you specified)

[1] Specify freesurfer directory, subject, and surface type
> "/my_project/derivatives/freesurfer/"
> "sub-01"
> "pial" (or could be "inflated")
This gives us the location of the freesurfer file which has the coordinates of every vertex in the mesh, as well as which vertices go together to form the face. This freesurfer file is currently in a binarised format - which takes lower memory, but cannot be read as text. 
[2] Use freesurfer function "mris_convert


# Experimental way to view surfaces (without pycortex)
# Stages:
# [1] Specify the mesh, created by freesurfer you want to use
# -> e.g "pial", or "inflated" 
# [2] Convert from freesurfer file to asc
# -> mris_convert  asc 
# -> rename .asc as .srf file
# [3] Convert from .srf to .ply (using brainder scripts)
# [4] Load the .ply file in meshlab


## In active development
Stuff may well change. 

## Installation
Standard git install 
Then setup using `python setup.py develop`
