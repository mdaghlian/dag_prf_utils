#!/usr/bin/env python
#$ -j Y
#$ -cwd
#$ -V

import numpy as np  
import nibabel as nb
import os
import getopt
import linescanning.utils as lsutils
opj = os.path.join

from amb_scripts.load_saved_info import *
from nibabel.freesurfer.io import read_morph_data, write_morph_data
import matplotlib as mpl
import matplotlib.pyplot as plt

from dag_prf_utils.utils import *
project_dir = os.environ.get('DIR_PROJECTS')
derivatives_dir = '/data1/projects/dumoulinlab/Lab_members/Marcus/projects/amblyopia_emc/derivatives'


class FSMaker(object):
    '''Used to make a freesurfer file, and view a surface in freesurfer. 
    One of many options for surface plotting. 
    Will create a curv file in subjects freesurfer dir, and load it a specific colormap 
    saved as the relevant command
    '''
    def __init__(self, sub, fs_dir):
        self.sub = sub        
        self.fs_dir = fs_dir        # Where the freesurfer files are        
        self.sub_surf_dir = opj(fs_dir, sub, 'surf')
        self.custom_surf_dir = opj(self.sub_surf_dir, 'custom')
        if not os.path.exists(self.custom_surf_dir):
            os.mkdir(self.custom_surf_dir)        
        n_vx = dag_load_nverts(self.sub, self.fs_dir)
        self.n_vx = {'lh':n_vx[0], 'rh':n_vx[1]}
        self.overlay_str = {}
        self.open_surf_cmds = {}

    def add_surface(self, data, surf_name, **kwargs):
        '''
        See dag_calculate_rgb_vals...
        data            np.ndarray      What are we plotting...
        surf_name       str             what are we calling the file

        '''

        data_mask = kwargs.get('data_mask', np.ones_like(data, dtype=bool))
        # Load colormap properties: (cmap, vmin, vmax)
        cmap = kwargs.get('cmap', 'viridis')    
        vmin = kwargs.get('vmin', np.percentile(data[data_mask], 10))
        vmax = kwargs.get('vmax', np.percentile(data[data_mask], 90))
        cmap_nsteps = kwargs.get('cmap_nsteps', 10)

        data_masked = np.zeros_like(data, dtype=float)
        data_masked[data_mask] = data[data_mask]
        exclude_min_val = vmin - 1
        data_masked[~data_mask] = exclude_min_val

        # SAVE masked data AS A CURVE FILE
        lh_masked_param = data_masked[:self.n_vx['lh']]
        rh_masked_param = data_masked[self.n_vx['lh']:]

        # now save results as a curve file
        print(f'Saving {surf_name} in {self.custom_surf_dir}')

        write_morph_data(opj(self.custom_surf_dir, f'lh.{surf_name}'),lh_masked_param)
        write_morph_data(opj(self.custom_surf_dir, f'rh.{surf_name}'),rh_masked_param)        
        
        # Make custom overlay:
        # value - rgb triple...
        fv_param_steps = np.linspace(vmin, vmax, cmap_nsteps)
        fv_color_steps = np.linspace(0,1, cmap_nsteps)
        fv_cmap = mpl.cm.__dict__[cmap]
        
        ## make colorbar - uncomment to save a png of the color bar...
        # cb_cmap = mpl.cm.__dict__[cmap] 
        # cb_norm = mpl.colors.Normalize()
        # cb_norm.vmin = vmin
        # cb_norm.vmax = vmax
        # plt.close('all')
        # plt.colorbar(mpl.cm.ScalarMappable(norm=cb_norm, cmap=cb_cmap))
        # col_bar = plt.gcf()
        # col_bar.savefig(opj(self.sub_surf_dir, f'lh.{surf_name}_colorbar.png'))

        overlay_custom_str = 'overlay_custom='
        for i, fv_param in enumerate(fv_param_steps):
            this_col_triple = fv_cmap(fv_color_steps[i])
            this_str = f'{float(fv_param):.2f},{int(this_col_triple[0]*255)},{int(this_col_triple[1]*255)},{int(this_col_triple[2]*255)},'

            # print(this_str)
            overlay_custom_str += this_str    
        
        print('Custom overlay string saved here: (self.overlay_str[surf_name])')
        self.overlay_str[surf_name] = overlay_custom_str
    
    def open_fs_surface(self, surf_name, mesh='inflated'):
        # surf name - which surface to load...
        # mesh -> loading inflated? pial? etc.
        os.chdir(self.sub_surf_dir) # move to freeview dir        
        fview_cmd = self.save_fs_cmd(surf_name=surf_name, mesh=mesh)
        os.system(fview_cmd)        

    def save_fs_cmd(self, surf_name, mesh='inflated'):
        lh_surf_path = opj(self.custom_surf_dir, f'lh.{surf_name}')
        rf_surf_path = opj(self.custom_surf_dir, f'rh.{surf_name}')

        fview_cmd = f'''freeview -f lh.{mesh}:overlay={lh_surf_path}:{self.overlay_str[surf_name]} rh.{mesh}:overlay={rf_surf_path}:{self.overlay_str[surf_name]}'''
        dag_str2file(filename=opj(self.custom_surf_dir, f'{surf_name}_cmd.txt'),txt=fview_cmd)
        return fview_cmd

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


