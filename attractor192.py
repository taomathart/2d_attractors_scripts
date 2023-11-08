from matplotlib import cm
import numpy as np
import io
import pandas as pd
import datashader as ds
from datashader import transfer_functions as tf
from PIL import Image
from numba import jit
import time

import cmasher as cmr

@jit(nopython=True)
def getX(coef,x,y,i):
    return coef[0]*x*np.cos(y*coef[1])*np.sin(coef[2])

@jit(nopython=True)
def getY(coef,x,y,i):
    return coef[3]*x*np.sin(y*coef[4])*np.cos(coef[5])
    
@jit(nopython=True)
def generatePoints(diter,coefs,x0=None,y0=None):
    print(coefs)
    print('Generating points')

    x = [0.1] if x0 is None else [x0]
    y = [0.1] if y0 is None else [y0]

    for i in range(diter):
        x.append(getX(coefs,x[i-1],y[i-1],i))
        y.append(getY(coefs,x[i-1],y[i-1],i))
    print('Iteration done')

    return x,y

start = time.process_time()
iterations = 20000000 #number of points, you can change this value
sizepx = 2000   #resolution , you can change this value

colormap = cm.get_cmap("cmr.ocean")

background = 'black'
coefs = [-1.5828203013256454,0.22061769652760566,2.20533304283353,2.1715207479969507,1.4303895527721684,0.7490259126975705]

x,y = generatePoints(int(iterations),np.array(coefs))
df = pd.DataFrame(dict(x=x,y=y))

cvs = ds.Canvas(plot_width = int(sizepx), plot_height = int(sizepx), x_range=(-3,3), y_range=(-3,3))
agg = cvs.points(df, 'x', 'y')
img = tf.set_background(tf.shade(agg, cmap = colormap),background)
buf = io.BytesIO()
img.to_pil().save(buf, format='PNG')
image = Image.open(buf)
image.save("attractor192.png","PNG")
buf.close()
image.close()
img.close()
print('DONE')
print('Time process : '+str(round(time.process_time() - start,2))+' seconds')