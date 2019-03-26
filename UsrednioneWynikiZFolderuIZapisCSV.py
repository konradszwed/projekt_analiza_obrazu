#!/usr/bin/env python
# coding: utf-8

# In[160]:


#Konrad Szwed projekt 0.1

#Bibliografia:
#https://dzone.com/articles/image-data-analysis-using-numpy-amp-opencv


# In[258]:


import glob
import os
import numpy as np
import imageio
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import datetime
import csv


# In[259]:


#ŚCIEŻKI DOSTĘPU
#WPISZ SCIEZKE DO FOLDERU W KTORYM ZBADAMY PLIKI GRAFICZNE (zostaw to r''' i ''' na końcu)
#domyślnie: \Users\rooni\Desktop\projekt_analiza_obrazu\baza
#########################################################################
user_path=r'''C:\Users\rooni\Desktop\PHOTOSHOP'''
#########################################################################
stats_path=r'''C:\Users\rooni\Desktop\projekt_analiza_obrazu\stats'''


# In[260]:


#pobieranie stricte obrazow z folderu
path, dirs, files = next(os.walk(user_path))

print("W folderze jest :", len(files) ,"plikow")
files = [ fi for fi in files if fi.endswith(".jpg") ]
file_count = len(files)
print("W tym :", len(files) ,"plików jpg")


# In[267]:


def imgstats(file):
    pic = imageio.imread(file)
    img_stats=[]
    global i
    img_stats.append(i)
    img_stats.append(file)
    img_stats.append(type(pic))
    img_stats.append(pic.shape[0])
    img_stats.append(pic.shape[1])
    img_stats.append(pic.ndim)
    img_stats.append(pic.size)
    img_stats.append(pic.max())
    img_stats.append(pic.min())
    img_stats.append(pic[ 100, 50, 0])
    img_stats.append(pic[ 100, 50, 1])
    img_stats.append(pic[ 100, 50, 2])
    return img_stats


# In[268]:


Titles=['Index','File name','Type of the image','Image Height','Image Width','Dimension of Image','Image size','Maximum RGB','Minimum RGB','R channel','G channel','B channel','Baza wygenerowana przez Konrad Szwed dnia {} | Zawiera {} elementow'.format(datetime.datetime.today(),file_count)]


# In[275]:


#Sprawdzenie poprzedniego pliku ze statystykami
#path_oldstat='./stats/'
path_oldstat=stats_path
if os.path.isfile(path_oldstat+'stats.csv') :
    if os.path.isfile(path_oldstat+'old_stats.csv') :    
        os.unlink(path_oldstat+'old_stats.csv')
    os.rename(path_oldstat+'stats.csv', path_oldstat+'old_stats.csv')
    print('Utworzono backup poprzedniej bazy danych w folderze {},pod nazwą {}'.format(path_oldstat, os.listdir(path_oldstat)))
    
#Tworzenie csv
with open(stats_path+'\stats.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(Titles)
    #file_count ale do testow ustawcie mniejsza liczbe
    for i in range(file_count):
        file=user_path+'\{}'.format(files[i])
        filewriter.writerow(imgstats(file))
        
#print(imgstats('baza\{}'.format(files[0])))


# In[323]:


def srednia(stats):
    return('Image Height:{}\n Image Width:{}\n Dimension of Image:{}\n Image size:{}\n Maximum RGB:{}\n Minimum RGB:{}\n R channel:{}\n G channel:{}\n B channel:{}'.format(stats[0],stats[1],stats[2],stats[3],stats[4],stats[5],stats[6],stats[7],stats[8]))


# In[324]:


#Statystyki uśrednione
stats=[]
for i in range(9):
    stats.append(0)
for i in range(file_count):
    file=user_path+'\{}'.format(files[i])
    stats2=imgstats(file)  
    for n in range(9):
        stats[n]+=stats2[n+3]
        
for i in range(9):
    stats[i]=stats[i]/file_count
    
print("Wyniki średnie:")    
print(srednia(stats))


# In[ ]:




