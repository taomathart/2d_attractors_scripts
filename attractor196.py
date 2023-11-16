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
    val1 = coef[1]*np.cos(x)
    if(val1==0): val1=0.00001
    val3 = coef[3]*np.sin(y)
    if(val3==0): val3=0.00001
    return np.sin(coef[0]*y) / val1 + \
           np.cos(coef[2]*x) / val3

@jit(nopython=True)
def getY(coef,x,y,i):
    val5 = coef[5]*np.sin(y)
    if(val5==0): val5=0.00001
    val7 = coef[7]*np.cos(x)
    if(val7==0): val7=0.00001
    return np.cos(coef[4]*x) / val5 - \
           np.sin(coef[6]*y) / val7
    
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
iterations = 20000000 #number of points, you can change this value 27.62
sizepx = 2000   #resolution , you can change this value

colormap = cm.get_cmap("cmr.jungle")

background = 'black'
coefs = [-1.7798400717729943,1.243286618030763,2.739940796925918,-1.5133473580555163,1.1401215305396466,-1.764327624098752,-1.3192265229389357,1.52977586616489]

x,y = generatePoints(int(iterations),np.array(coefs))
df = pd.DataFrame(dict(x=x,y=y))

cvs = ds.Canvas(plot_width = int(sizepx), plot_height = int(sizepx), x_range=(-3,3), y_range=(-3,3))
agg = cvs.points(df, 'x', 'y')
img = tf.set_background(tf.shade(agg, cmap = colormap),background)
buf = io.BytesIO()
img.to_pil().save(buf, format='PNG')
image = Image.open(buf)
image.save("attractor196.png","PNG")
buf.close()
image.close()
img.close()
print('DONE')
print('Time process : '+str(round(time.process_time() - start,2))+' seconds')