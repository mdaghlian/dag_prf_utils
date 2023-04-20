#!/usr/bin/env python
#$ -j Y
#$ -cwd
#$ -V

import numpy as np  
import nibabel as nb
import os
import getopt
opj = os.path.join

from amb_scripts.load_saved_info import *
from nibabel.freesurfer.io import read_morph_data, write_morph_data
import matplotlib as mpl
import matplotlib.pyplot as plt

brainder_dir = os.environ.get('PATH_HOME')
derivatives_dir = '/data1/projects/dumoulinlab/Lab_members/Marcus/projects/amblyopia_emc/derivatives'

blender_dir = opj(derivatives_dir, 'blender_objs')
if not os.path.exists(blender_dir):
    os.mkdir(blender_dir)


def main(argv):

    """
---------------------------------------------------------------------------------------------------
WRITE A BLENDER FILE
Made by MD 

Arguments:
    -s|--sub    <sub number>        number of subject's FreeSurfer directory from which you can 
                                    omit "sub-" (e.g.,for "sub-001", enter "001").
    --model     model name
    -t|--task   <task name>         name of the experiment performed (e.g., "LE", "RE")
    -p|--param  <parameter to plot> e.g., polar angle
    --roi_fit   sometimes we fit only a subset of voxels, ("e.g. V1_exvivo")
    --rsq_th    rsq threshold
    --ecc_th    ecc threshold
Options:                                  
    -v|--verbose    print some stuff to a log-file
    --overwrite     If specified, we'll overwrite existing Gaussian parameters. If not, we'll look
                    for a file with ['model-gauss', 'stage-iter', 'params.npy'] in *outputdir* and,
                    if it exists, inject it in the normalization model (if `model=norm`)  

Example:


---------------------------------------------------------------------------------------------------
"""

    sub         = None
    task        = None
    param       = 'pol'
    model       = None
    ses         = 'ses-1'
    roi_fit     = 'all'
    rsq_th      = 0.1
    ecc_th      = 5
    overwrite   = True
    under_surf  = 'inflated'
    scr_shot    = False
    verbose     = True

    try:
        opts = getopt.getopt(argv,"h:s:n:t:m:v:",["help", "sub=", "ses=", "task=", "param=", "model=", 
                                              "rsq_th=", "roi_fit=", "ecc_th=", "under_surf=", "scr_shot","verbose", "overwrite"])[0]
    except getopt.GetoptError:
        print("ERROR while reading arguments; did you specify an illegal argument?")
        print(main.__doc__)
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(main.__doc__)
            sys.exit()
        elif opt in ("-s", "--sub"):
            sub = hyphen_parse('sub', arg)
        elif opt in ("-n", "--ses"):
            ses = hyphen_parse('ses', str(arg))
        elif opt in ("-t", "--task"):
            task = arg# hyphen_parse('task', arg)
        elif opt in ("-p", "--param"):
            param = arg
        elif opt in ("--rsq_th"):
            rsq_th = float(arg)
        elif opt in ("--ecc_th"):
            ecc_th = float(arg)            
        elif opt in ("--roi_fit"):
            roi_fit = arg
        elif opt in ("--model"):
            model = arg
        elif opt in ("--under_surf"):
            under_surf = arg            
        elif opt in ("--overwrite"):
            overwrite = True
        elif opt in ("--scr_shot"):
            scr_shot = True            
        elif opt in ("-v", "--verbose"):
            verbose = True
        else:
            print(opt)

            sys.exit()

    if len(argv) < 2:
        print(main.__doc__)
        sys.exit()

    # Check subject path for matching surface:
    path_to_sub_surf = opj(derivatives_dir, 'freesurfer', sub, 'surf')
    surf_name = f'{sub}_{param}_{task}_{model}'
    if verbose:
        print(f'Making new surface files for {sub}..')
    
    prf_params  = amb_load_prf_params(sub=sub, task_list=task, model_list=model, roi_fit=roi_fit)[task][model]
    r2_mask = prf_params[:,-1] > rsq_th

    if 'pRF' in task:
        ecc = np.sqrt(prf_params[:,0]**2 + prf_params[:,1]**2)
        pol = np.arctan2(prf_params[:,1], prf_params[:,0])
        ecc_mask = ecc < ecc_th
        total_mask = r2_mask * ecc_mask
    else:
        total_mask = r2_mask


    param_dict = print_p()[model]
    if param=='pol':
        masked_param = np.where(total_mask, pol, -10)
        fv_vmin = -3.14
        fv_vmax = 3.14
        fv_cmap_name = 'hsv'
        
    elif param=='ecc':
        masked_param = np.where(total_mask, ecc, -1)
        fv_vmin = 0
        fv_vmax = ecc_th
        fv_cmap_name = 'cool'
    else:
        masked_param = np.zeros_like(total_mask, dtype=float)
        masked_param[total_mask] =  prf_params[total_mask,param_dict[param]]
        fv_vmin = np.min(prf_params[total_mask,param_dict[param]])
        fv_vmax = np.max(prf_params[total_mask,param_dict[param]])
        exclude_min_val = fv_vmin - 1
        masked_param[~total_mask] = exclude_min_val
        fv_cmap_name = 'cool'

    # SAVE RESULTS AS A CURVE FILE
    lh_c = read_morph_data(opj(path_to_sub_surf,'lh.curv'))
    lh_masked_param = masked_param[:lh_c.shape[0]]
    rh_masked_param = masked_param[lh_c.shape[0]:]

    # now save results as a curve file
    write_morph_data(opj(path_to_sub_surf, f'lh.{surf_name}'),lh_masked_param)
    write_morph_data(opj(path_to_sub_surf, f'rh.{surf_name}'),rh_masked_param)        
    
    # Make custom overlay:
    # value - rgb triple...
    fv_nsteps = 10
    fv_param_steps = np.linspace(fv_vmin, fv_vmax, fv_nsteps)
    fv_color_steps = np.linspace(0,1, fv_nsteps)
    fv_cmap = mpl.cm.__dict__[fv_cmap_name]
    
    # Make pol colorbar
    # make colorbar
    # x_grid = np.linspace(-1,1,10)
    # y_grid = np.linspace(-1,1,10)
    # x_grid,y_grid = np.meshgrid(x_grid,y_grid)
    # ang_grid = np.arctan2(y_grid, x_grid)
    # plt.imshow(ang_grid, cmap='hsv', vmin=fv_vmin, vmax=fv_vmax)
    # plt.show(block=False)

    overlay_custom_str = 'overlay_custom='
    for i, fv_param in enumerate(fv_param_steps):
        this_col_triple = fv_cmap(fv_color_steps[i])
        this_str = f'{float(fv_param):.2f},{int(this_col_triple[0]*255)},{int(this_col_triple[1]*255)},{int(this_col_triple[2]*255)},'

        # print(this_str)

        overlay_custom_str += this_str    
    # Otherwise just load the surface...
    os.chdir(path_to_sub_surf) # move to freeview dir
    # under_surf = 'inflated' # inflated
    # if 
    fview_cmd = f'''freeview  -f lh.{under_surf}:overlay=lh.{surf_name}:{overlay_custom_str} rh.{under_surf}:overlay=rh.{surf_name}:{overlay_custom_str} & '''
    os.system(fview_cmd)
    # os.system('freeview ')

'''
[-3.14,255,  0,  0,
 -2.65,255,255,  0,
 -2.09,  0,128,  0,
 -1.75,  0,255,255,
 -1.05,  0,  0,255,
 - 0.5,238,130,238,
     0,255,0,0,
   0.5,255,255,0,1.05,0,128,0,1.57,0,255,255,2.09,0,0,255,2.65,238,130,238,3.14,255,0,0]
'''

# *************
if __name__ == "__main__":
    main(sys.argv[1:])


