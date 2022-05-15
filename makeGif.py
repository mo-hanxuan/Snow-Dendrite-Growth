from genericpath import exists
from moviepy.editor import ImageSequenceClip
import os
path = input("\033[35;1m please give the folder path where your pictures stored at. path = \033[0m")
path += "/"
img_names = []
for i in range(0, 6336, 32):
    file = path + str(i) + '.png'
    if os.path.exists(file):
        img_names.append(file)
clip = ImageSequenceClip(img_names, fps=24)
clip.write_gif(path + 'snow_growing.gif')