# Check for requirements:
import os
import subprocess
# [1] Freesurfer, freeview
# fs_cmd_list = ['freeview']
# for cmd in fs_cmd_list:
#     cmd_out = subprocess.getstatusoutput(f"command -v {cmd}")[1]
#     if cmd_out=='':
#         print(f'Could not find path for {cmd}, is freesurfer accessible from here?')

# ************************** SPECIFY FS_LICENSE HERE **************************
# os.environ['FS_LICENSE'] = '/data1/projects/dumoulinlab/Lab_members/Marcus/programs/linescanning/misc/license.txt'
# if 'FS_LICENSE' in os.environ.keys():
#     if not os.path.exists(os.environ['FS_LICENSE']):
#         print('Could not find FS_LICENSE, set using os.environ above')
# else:
#     print('Could not find FS_LICENSE')
#     print('Uncomment line below and specify path to FS_LICENSE')

# # [2] Nibabel
# try:
#     from nibabel.freesurfer.io import write_morph_data        
# except ImportError:
#     print('Error importing nibabel... Not a problem unless you want to use FSMaker')

# [3] pycortex
# try: 
#     import cortex 
# except ImportError:
#     print('Error importing pycortex... Not a problem unless you want to use pycortex stuff')



# ************************** SPECIFY BLENDER PATH HERE **************************
# os.environ['BLENDER'] = '/data1/projects/dumoulinlab/Lab_members/Marcus/programs/blender-3.5.1-linux-x64/blender'
# if not 'BLENDER' in os.environ.keys():
#     # Check for command:
#     blender_cmd = subprocess.getstatusoutput(f"command -v blender")[1]
#     if blender_cmd == '':
#         print('could not find blender command, specify in __init__ file')
#     else:
#         os.environ['BLENDER'] = blender_cmd
# elif not os.path.exists(os.environ['BLENDER']):    
#     print('Could not find blender, specify location in __init__ file')
#     print('only a problem if you want to use blender...')


# ************************** CHECK FOR SUBJECTS DIR **************************
# if "SUBJECTS_DIR" not in os.environ.keys():
#     print('SUBJECTS_DIR not found in os.environ')
#     print('Adding empty string...')
#     os.environ['SUBJECTS_DIR'] = ''
    