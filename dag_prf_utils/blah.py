


import matplotlib.pyplot as plt
from io import BytesIO
import IPython.display as display

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