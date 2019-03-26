#!/usr/bin/env python
# coding: utf-8

# In[32]:


#Konrad Szwed projekt 0.1

#Bibliografia:
#https://dzone.com/articles/image-data-analysis-using-numpy-amp-opencv


# In[64]:


import os
import numpy as np
import imageio
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[65]:


path, dirs, files = next(os.walk(r'''\Users\rooni\Desktop\projekt_analiza_obrazu\baza'''))
file_count = len(files)
print("W bazie jest :", file_count ," obrazów")


# In[66]:


#ścieżka do bazy względem pliku programu
file='baza\{}'.format(files[0])
pic = imageio.imread(file)
plt.figure(figsize = (15,15))
plt.imshow(pic)


# In[76]:


print('Type of the image: ' , type(pic))
print('Shape of the image: {}'.format(pic.shape))
print(f'Image Hight: {pic.shape[0]} pixels')
print(f'Image Width: {pic.shape[1]} pixels')
print('Dimension of Image: {}'.format(pic.ndim))
print('Image size: {}'.format(pic.size))
print('Maximum RGB value in this image: {}'.format(pic.max()))
print('Minimum RGB value in this image: {}'.format(pic.min()))
print('Value of only R channel: {}'.format(pic[ 100, 50, 0]))
print('Value of only G channel: {}'.format(pic[ 100, 50, 1]))
print('Value of only B channel: {}'.format(pic[ 100, 50, 2]))


# In[68]:


#channel Red
#plt.title('R channel')
#plt.ylabel('Height {}'.format(pic.shape[0]))
#plt.xlabel('Width {}'.format(pic.shape[1]))
#plt.imshow(pic[ : , : , 0])
#plt.show()

#channel Green
#plt.title('G channel')
#plt.ylabel('Height {}'.format(pic.shape[0]))
#plt.xlabel('Width {}'.format(pic.shape[1]))
#plt.imshow(pic[ : , : , 1])
#plt.show()

#channel Blue
#plt.title('B channel')
#plt.ylabel('Height {}'.format(pic.shape[0]))
#plt.xlabel('Width {}'.format(pic.shape[1]))
#plt.imshow(pic[ : , : , 2])
#plt.show()
print('Kanały RGB')
fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(15,4))  # 1 row, 3 columns
ax1.imshow(pic[ : , : , 0])
ax2.imshow(pic[ : , : , 1])
ax3.imshow(pic[ : , : , 2])
plt.show()


# In[71]:


print("Kanały RGB \n100% danego koloru")

fig, ax = plt.subplots(nrows = 1, ncols=3, figsize=(15,5))
for c, ax in zip(range(3), ax):
    # create zero matrix
    split_img = np.zeros(pic.shape, dtype="uint8") # 'dtype' by default: 'numpy.float64'
    # assing each channel 
    split_img[ :, :, c] = pic[ :, :, c]
    # display each channel
    ax.imshow(split_img)


# In[ ]:




