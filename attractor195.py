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
    val1 = np.cos(coef[1]*x)
    if(val1==0): val1=0.00001
    val3 = np.sin(coef[3]*y)
    if(val3==0): val3=0.00001
    return np.sin(coef[0]*y) / val1 + \
           np.cos(coef[2]*x) / val3

@jit(nopython=True)
def getY(coef,x,y,i):
    val5 = np.sin(coef[5]*y)
    if(val5==0): val5=0.00001
    val7 = np.cos(coef[7]*x)
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
iterations = 20000000 #number of points, you can change this value
sizepx = 2000   #resolution , you can change this value

colormap = cm.get_cmap("cmr.fall")

background = 'black'
coefs = [1.0953385196052876,2.489891399555237,-0.16325236623924022,0.9766760039942897,-0.49753657286398134,-2.0116003703707004,1.9936042805204934,-0.8728159212775886]

x,y = generatePoints(int(iterations),np.array(coefs))
df = pd.DataFrame(dict(x=x,y=y))

cvs = ds.Canvas(plot_width = int(sizepx), plot_height = int(sizepx), x_range=(-3,3), y_range=(-3,3))
agg = cvs.points(df, 'x', 'y')
img = tf.set_background(tf.shade(agg, cmap = colormap),background)
buf = io.BytesIO()
img.to_pil().save(buf, format='PNG')
image = Image.open(buf)
image.save("attractor195.png","PNG")
buf.close()
image.close()
img.close()
print('DONE')
print('Time process : '+str(round(time.process_time() - start,2))+' seconds')