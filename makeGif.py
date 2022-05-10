from moviepy.editor import ImageSequenceClip
path = './pictures/makeGif/'
img_names = [path + str(i) + '.png' for i in range(0, 6080, 32)]
clip = ImageSequenceClip(img_names, fps=24)
clip.write_gif(path + 'snow_512x512.gif')