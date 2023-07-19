from matplotlib import cm
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
def getX(coef,x,y,i):
    return coef[0] + coef[1]*x + coef[2]*y + \
           coef[3]*np.abs(x)**coef[4] + coef[5]*np.abs(y)**coef[6]

@jit(nopython=True)
def getY(coef,x,y,i):
    return coef[7] + coef[8]*x + coef[9]*y + \
           coef[10]*np.abs(x)**coef[11] + coef[12]*np.abs(y)**coef[13]
    
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

colormap = cm.get_cmap("cmr.savanna")

background = 'black'
coefs = [-0.48740689562295136,-0.8528061251238543,-0.45885221751017546,-0.10218251722719396,-0.9715222322886534,-0.3304549925038438,-0.7852784383744362,-0.6275130424067721,0.6153228297519031,-0.8105783769155197,0.7375865384349296,0.11039386017283292,-0.5875354654272087,0.5413734176261886]

x,y = generatePoints(int(iterations),np.array(coefs))
df = pd.DataFrame(dict(x=x,y=y))

cvs = ds.Canvas(plot_width = int(sizepx), plot_height = int(sizepx), x_range=(float(-5),float(5)), y_range=(float(-5),float(5)))
agg = cvs.points(df, 'x', 'y')
img = tf.set_background(tf.shade(agg, cmap = colormap),background)
buf = io.BytesIO()
img.to_pil().save(buf, format='PNG')
image = Image.open(buf)
image.save("attractor188.png","PNG")
buf.close()
image.close()
img.close()
print('DONE')
print('Time process : '+str(round(time.process_time() - start,2))+' seconds')