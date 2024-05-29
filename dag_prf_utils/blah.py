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