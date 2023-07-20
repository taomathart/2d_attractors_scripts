from matplotlib import cm
import numpy as np
import io
import pandas as pd
import datashader as ds
from datashader import transfer_functions as tf
from PIL import Image
from numba import jit
import time

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

colormap = cm.get_cmap("cividis")

background = 'black'
coefs = [-0.9686548854587009,-0.046834933849511984,-0.5722870548537158,-0.7864582926687822,-0.33707478656612366,-0.9259589163453594,-0.934197213134456,-0.14691369258431042,0.4766936863414355,0.075711487461392,0.8279470515642409,0.21570876674596962,-0.2664733769768828,0.3625690089866971]

x,y = generatePoints(int(iterations),np.array(coefs))
df = pd.DataFrame(dict(x=x,y=y))

cvs = ds.Canvas(plot_width = int(sizepx), plot_height = int(sizepx), x_range=(float(-7),float(2)), y_range=(float(-3),float(2)))
agg = cvs.points(df, 'x', 'y')
img = tf.set_background(tf.shade(agg, cmap = colormap),background)
buf = io.BytesIO()
img.to_pil().save(buf, format='PNG')
image = Image.open(buf)
image.save("attractor189.png","PNG")
buf.close()
image.close()
img.close()
print('DONE')
print('Time process : '+str(round(time.process_time() - start,2))+' seconds')