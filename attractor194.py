import numpy as np
import io
import pandas as pd
import datashader as ds
from datashader import transfer_functions as tf
from PIL import Image
from numba import jit
import time
import cmocean

@jit(nopython=True)
def getX(coef,x,y,i):
    val0 = coef[0]
    if(val0 == 0): val0 = 0.00001
    val1 = coef[1]
    if(val1 == 0): val1 = 0.00001
    return np.sin((x*y)/val0)*y + np.cos((x*y)/val1)

@jit(nopython=True)
def getY(coef,x,y,i):
    val2 = coef[2]
    if(val2 == 0): val2 = 0.00001
    val3 = coef[3]
    if(val3 == 0): val3 = 0.00001
    return np.cos((y*x)/val2)*y + np.sin((y*x)/val3)
    
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

colormap = cmocean.cm.cmap_d.get("matter_r")

background = 'black'
coefs = [-1.36930665506933,0.0367718033488007,-1.3570317968166254,-0.017590356290273945]

x,y = generatePoints(int(iterations),np.array(coefs))
df = pd.DataFrame(dict(x=x,y=y))

cvs = ds.Canvas(plot_width = int(sizepx), plot_height = int(sizepx), x_range=(-3.5,3.5), y_range=(-3.5,3.5))
agg = cvs.points(df, 'x', 'y')
img = tf.set_background(tf.shade(agg, cmap = colormap),background)
buf = io.BytesIO()
img.to_pil().save(buf, format='PNG')
image = Image.open(buf)
image  = image.rotate(90)
image.save("attractor194.png","PNG")
buf.close()
image.close()
img.close()
print('DONE')
print('Time process : '+str(round(time.process_time() - start,2))+' seconds')