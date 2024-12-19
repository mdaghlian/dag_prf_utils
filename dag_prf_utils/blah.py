


import matplotlib.pyplot as plt
from io import BytesIO
import IPython.display as display
# plt.hist(edge_lengths, bins=20)
# plt.axvline(m_edge_length, c='r')
# plt.axvline(m_edge_length + edge_factor*std_edge_length, c='r')
# bloop

# pts = np.vstack([fake_flat['x'],fake_flat['y'], fake_flat['z']]).T
# PREVIOUS CUTTING STRATEGIES...
# outer_vx = ConvexHull(pts[:,:2]).vertices
# plt.figure()
# plt.scatter(
#     pts[:,0], pts[:,1], c=sphere_mesh_info['x']
# )
# plt.scatter(
#     pts[outer_vx,0], pts[outer_vx,1], c='r'
# )
# # bloop
# in_face_x = {} 
# for face_x in ['i', 'j', 'k']:
#     in_face_x[face_x] = np.isin(sphere_mesh_info[face_x], outer_vx) * 1.0    

# # remove faces with 2+ vx in the cut
# f_w_23cutvx = (in_face_x['i'] + in_face_x['j'] + in_face_x['k']) >= 2
# print(f'Faces with 2+ vx in the cut: {f_w_23cutvx.sum()}')
# fake_flat['faces']  = sphere_mesh_info['faces'][~f_w_23cutvx,:]
# fake_flat['i']      = sphere_mesh_info['i'][~f_w_23cutvx]
# fake_flat['j']      = sphere_mesh_info['j'][~f_w_23cutvx]
# fake_flat['k']      = sphere_mesh_info['k'][~f_w_23cutvx]


# # Now find those outer vertices which appear in 2 faces
# flat_faces = fake_flat['faces'].ravel()
# vx_counts = np.sum(flat_faces[:, None]==outer_vx, axis=0)
# print(vx_counts)

# vx_in2face = np.where(vx_counts==2)[0]
# faces2remove = []
# for i_outvx in vx_in2face:
#     this_vx = outer_vx[i_outvx]
#     # find it ...

#     faces2remove.append(
#         np.where(np.sum(fake_flat['faces'] == this_vx, axis=1) > 0)[0][0]
#     )
# # Remove the face
# if faces2remove is not []:
#     fake_flat['faces'] = np.delete(fake_flat['faces'], faces2remove, axis=0)
#     fake_flat['i'] = np.delete(fake_flat['i'], faces2remove)
#     fake_flat['j'] = np.delete(fake_flat['j'], faces2remove)
#     fake_flat['k'] = np.delete(fake_flat['k'], faces2remove)

# *********************************************
# Now find those outer vertices which appear in 2 faces
# flat_faces = fake_flat['faces'].ravel()
# vx_counts = np.sum(flat_faces[:, None]==outer_vx, axis=0)
# polys = fake_flat['faces']  
# Assuming fig1, fig2, fig3, fig4 are your figures
def fig_to_img(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf


buf1 = fig_to_img(prf_obj[sub].prf_obj[f'{task1}f_G'].prf_ts_plot(idx))
buf2 = fig_to_img(prf_obj[sub].prf_obj[f'{task1}f_N'].prf_ts_plot(idx))
buf3 = fig_to_img(prf_obj[sub].prf_obj[f'{task2}f_G'].prf_ts_plot(idx))
buf4 = fig_to_img(prf_obj[sub].prf_obj[f'{task2}f_N'].prf_ts_plot(idx))
buf_m = []
for m in ['G', 'N']:
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # Do an arrow plot
    arr_mask = np.zeros_like(all_mask)
    arr_mask[idx] = True
    prf_obj[sub].arrow(
        f'{task1}f_{m}', f'{task2}f_{m}', th={'roi': arr_mask}, ax=ax, 
        arrow_col=mod_cols['gauss'] if m=='G' else mod_cols['norm'],
        old_col='g', new_col='r',
        do_scatter=True, 
    )
    buf_m.append(fig_to_img(fig))

fig, axes = plt.subplots(3, 2, figsize=(15, 10))
img_label = [f'{task1}f_G', f'{task2}f_G', f'{task1}f_N', f'{task2}f_N']
images = [buf1, buf2, buf3, buf4]
axes = axes.flatten()
for i in range(4):
    axes[i].imshow(plt.imread(images[i]))
    axes[i].set_title(img_label[i])
    axes[i].axis('off')

axes[-2].imshow(plt.imread(buf_m[0]))
axes[-2].axis('off')
axes[-1].imshow(plt.imread(buf_m[1]))
axes[-1].axis('off')





# ******************************************************************************************
# ******************************************************************************************
# ******************************************************************************************
'''
for highest movie quality (i.e., animated html)
'''
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
fig = plt.gcf()
def fig_updater(frame):
    '''
    Do something which depends on the frame
    '''
    fig.suptitle(frame)
# Hi resolution (save each one as an html)
with plt.rc_context({'animation.frame_format': 'svg'}):
    ani = FuncAnimation(
        fig, fig_updater, 
        frames=np.arange(0, 50),
        interval=50 # speed ... (time ms per frame)
        )
    # save as html
    with open('animation.html', 'w') as f:
        f.write(ani.to_jshtml())



# import svgwrite
# # Using wand to make animated svg
# for model in ['gauss', 'norm']:
#     for do_scot in [True, False]:
#         save_name = f'{model}_movie_scot-{do_scot}'
#         save_path = opj(fig_saver.path, save_name)
#         if not os.path.exists(save_path):
#             os.mkdir(save_path)
#         fig_paths = []
#         for time in range(50):
#             # model = 'norm'
#             # fig,ax = MOVIE_basic( # Norm bar
#             #     do_scot=do_scot, 
#             #     time_pt=time,            
#             #     model=model,
#             #     scot_kwargs= scot_kwargs[model],
#             #     )
#             fig_path = f"{save_path}/file{time:03}.svg"            
#             # fig.savefig(fig_path) 
#             # plt.close(fig)
#             fig_paths.append(fig_path)
        
#         duration = 100
#         num_frames = len(fig_paths)
        
#         ani_svg_path = opj(fig_saver.path, f'svgwriteer_{save_name}.svg')
#         # Create an svg document
#         dwg = svgwrite.Drawing(filename=ani_svg_path, size=("100%", "100%"))
#         # Define the animation element
#         animate = dwg.animate("indefinite", repeatCount="indefinite")
#         for i,svg_file in enumerate(fig_paths):
#             with open(svg_file, 'r') as f:
#                 frame_content = f.read()
#             begin_time = i * duration
#             end_time = (i+1) * duration
#             # create a new group for each frame 
#             text = dwg.text(frame_content, insert=(0,0), style='visibility;hidden;')
#             animate = dwg.animate(
#                 'visibility', from_='hidden', to='visible', 
#                 dur = f"{duration}ms",
#                 begin = f"{begin_time}ms",
#                 fill='freeze'
#                 )
#             text.add(animate)
#             dwg.add(text)
#         #     # add the frame content to the group
#         #     group.add(dwg.text(frame_content))
#         #     # add the group to the animation element
#         #     animate.add(group, begin=f'{i*100}ms')
#         # dwg.add(animate)
#         dwg.save()


#         break
#     break



            # We want the hemispheres to be on the same scale...
            # ... TODO, all kinds of orientation messing? 
            # if hemi == 'rh':
            #     # Save the bounding box for the largest dimension. We will match 
            # We want it to be roughly on the same scale as the inflated map
            # diff_x = pts[:,0].mean() - infl_x.mean()
            # diff_y = pts[:,1].mean() - infl_y.mean()
            # pts[:,0] -= diff_x
            # pts[:,1] -= diff_y
            # scale_x = (infl_x.max() - infl_x.min()) / (pts[:,0].max() - pts[:,0].min())
            # pts *= scale_x*3 # Meh seems nice enough

            # # OLD TOOO REMOVE!!!!
            # # FROM PYCORTEX IMPORT FLAT
            # # ORIGINAL FLIP X AND Y, THEN FLIP Y UPSIDE DOWN
            # # DOUBLE CHECK THE X,Y with SPHERE x,y
            # # corr_x = dag_get_corr(self.mesh_info['sphere'][hemi]['x'], pts[:,0])
            # # corr_y = dag_get_corr(self.mesh_info['sphere'][hemi]['y'], pts[:,1])
            # # if corr_x<0:
            # #     pts[:,0] *= -1
            # #     print('flipping x')
            # # if corr_y>0:
            # #     pts[:,1] *= -1
            # #     print('flipping y')
            # # if do_flip:
            # #     flat = pts[:, [1, 0, 2]] # Flip X and Y axis
            # #     # Flip Y axis upside down
            # #     flat[:, 1] = -flat[:, 1]
            # # else:
            # #     # bloop
            # #     flat = pts