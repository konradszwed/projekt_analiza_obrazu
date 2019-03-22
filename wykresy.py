#!/usr/bin/env python
# coding: utf-8

# In[24]:



import pylab
import scipy


# In[27]:


x = scipy.linspace(-2,2,1500)
y = scipy.sqrt(1-(abs(x)-1)**2)
z = -3*scipy.sqrt(1-(abs(x)/2)**0.5)
pylab.fill_between(x, y, color="fuchsia")
pylab.fill_between(x, z, color="fuchsia")
pylab.xlim([-2.5, 2.5])
pylab.text(0,-0.4, "Kocham Cię",fontsize=26, fontweight='bold', color="white", horizontalalignment='center')


# In[ ]:




