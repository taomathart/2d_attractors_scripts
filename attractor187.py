from matplotlib import cm
import colorcet as cc
import numpy as np
import io
import pandas as pd
import datashader as ds
from datashader import transfer_functions as tf
from PIL import Image, ImageOps
from numba import jit
import time
import cmasher as cmr

@jit(nopython=True)
def getX(coef,x,y,i):# x values, attactor 1
    x = getX1(coef,x,y,i)
    return (coef[0]+coef[1]*np.sin(x))*np.sin(y)

@jit(nopython=True)
def getY(coef,x,y,i):# y values, attactor 1
    y = getY1(coef,x,y,i)
    return (coef[2]+coef[3]*np.sin(x))*np.cos(y)

@jit(nopython=True)
def getX1(coef,x,y,i):# x values, attactor 2
    return np.exp(x*coef[0])*np.sin(y*coef[1])

@jit(nopython=True)
def getY1(coef,x,y,i):# y values, attactor 2
    return np.exp(x*coef[2])*np.cos(y*coef[3])
    
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
iterations = 2000000 #number of points, you can change this value (100.000.000 max (6000px resolution) for a good pc.. 16gb ram, proc i7 10700 takes aprox. 50 seconds)
sizepx = 2000   #resolution , you can change this value

colormap = cm.get_cmap("cmr.ocean")

background = 'black'
coefs = [2.899210625358158,-0.5578814184426273,0.9620240055847917,-0.8464045745142403] #constants used for the equations

x,y = generatePoints(int(iterations),np.array(coefs))
df = pd.DataFrame(dict(x=x,y=y))

cvs = ds.Canvas(plot_width = int(sizepx), plot_height = int(sizepx))
agg = cvs.points(df, 'x', 'y')
img = tf.set_background(tf.shade(agg, cmap = colormap),background)
buf = io.BytesIO()
img.to_pil().save(buf, format='PNG')
image = Image.open(buf)
image = ImageOps.expand(image,border=int(int(sizepx)/10),fill=background)
image.save("test.png","PNG") #you can change the image name
buf.close()
image.close()
img.close()
print('DONE')
print('Time process : '+str(round(time.process_time() - start,2))+' seconds')
