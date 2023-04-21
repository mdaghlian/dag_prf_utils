# dag_prf_utils repository

This repository is a collection of useful tools for analysing and visualising PRF data. It is designed to work with prfpy https://github.com/VU-Cog-Sci/prfpy, but with some of the tools are general purpose. 

## utils
Mainly generic tools - for searching for files, some mathematical stuff which comes up again and again in PRF analyses (converting between cartesian and polar coordinates etc; calculating rsq...).

In addition there are a couple of *prfpy specific tools*, i.e., Prf1T1M, etc., which are designed to load the outputs of prfpy models and hold them in easily manipulatable ways.

## plot functions
Several functions useful for plotting PRF properties around the visual field. All focus on matplotlib

## surface plotting and meshes
*Note that for many cases you will be better off using pycortex; this simply provides a couple of alternative methods, should you want to do that.*

I have created several tools for plotting data on the cortical surface. This will of course require that you have surface data (i.e., run freesurfer); and any data you want to plot will need to be in a vertex wise format. pycortex (https://github.com/gallantlab/pycortex) is a very powerful way of plotting information on the cortical surface and an excellent tool...but sometimes you just want to do simple stuff, and customize certain properties aspects (e.g., color maps etc.). Also it can be useful to have the cortical mesh in a generic format readable by any 3D rendering software (e.g., .ply), and in a way that you can send a single file anywhere, and it be instantly readable, with color and everything. 

To this end I have created a couple of different ways to visualize surface information. For all these methods it is important to think about:
* mesh: what is the mesh being used (it will mainly be 'inflated' or 'pial'). What is the actual shape that you want to look at 
* data: what do you want to plot on the mesh (e.g., polar angle, eccentricity. needs to be vertex wise). What colormap do you want to use. What mask do you want to use? (binary, only plot data for PRFs above a certain rsquared?; or do you want to weight the transparency of the data by a certain factor (again possibly rsquared),or some combination
* 'under_surface': if you are masking some of the data, what do you want to be shown in the gaps? Maybe the curvature of the cortex? Depth?

With this in mind here are a summary of some of the options. 

### Freeview based surface plotting
* requires freeview, and nibabel. 
* Specify the data, the mesh, and the mask (you can only use binary masking, the option for varying the transparency is not available here). 
* scripts will create a custom surf file, and the command (which contains the colormap info) to open it in freeview
* The colormap can be anything from matplotlib. Just specify the min and max values. 
* You can also specify the camera angle for when freeview opens, and ask it to automatically take a picture of the surface. This can be useful if you want to iterate through several subjects/surface plots and save the figures as pngs, but can't be bothered to sit and click again and again... 

### Generic .ply format
* requires freesurfer 
* Specify the data, the mesh, and the mask (including an option for variable transparency). 
* can create a single .ply file (per hemisphere), which contains all the information about the mesh (vx coordinates, face id); the data values for each vertex, and a color value for each vertex (determined by the data, and specified colormap). 
* This can be opened by most 3D viewing software (e.g., meshlab, blender)...

### Blender
* requires freesurfer and blender
* This is the most powerful approach and allows for a lot of customization (due to the blender api flexibility, which can be called via python)
* The script will load the inflated and pial mesh, with the option to slide between the 2 (i.e. customize how inflated you want the surface to be)
* You can load several colormaps at once, and flip between them 
* If you are feeling adventurous, you can even create an animation over time, (e.g., plot timecourse info on the surface). This is a bit experimental, and may take up a lot of data and computing power. I haven't fully explored it. 

For blender you can install using: https://www.blender.org/download/
For meshlab you can install using: https://www.meshlab.net/#download
## In active development
Stuff may well change. 

## Installation
Standard git install 
Then setup using `python setup.py develop`
