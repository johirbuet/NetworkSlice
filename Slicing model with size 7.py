#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This notebook will demonstrate slicing

import numpy as np
import skimage as ski
print(np.__file__)
print(np.__version__)
print(ski.__version__)
#from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import matplotlib.pyplot as plt

#from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import tensorflow as tf
from skimage.transform import resize
from graphviz import Graph, render
from utils.netviz import NetViz
from utils.mnistutil import MNISTUitl
from utils.sliceutil import Slice
sx = 28
sy = 28


# In[2]:


mn = MNISTUitl()
viz = NetViz()


# In[ ]:


X, Y, x, y = mn.getdata2(0,0,sx,sy)
na , xt, yt = mn.train4(X, Y, x,y,sx,sy,10,50)
nb , xt1, yt1 = na , xt, yt
nc , xt, yt = na , xt, yt
nd , xt1, yt1 = na , xt, yt
ne , xt, yt = na , xt, yt
nf , xt1, yt1 = na , xt, yt
ng , xt, yt = na , xt, yt
nh , xt1, yt1 = na , xt, yt
ni , xt, yt = na , xt, yt
nj , xt1, yt1 = na , xt, yt
hammingModel, xt1, yt1 = na , xt, yt

# In[ ]:


ya  = []
yb = []
yc  = []
yd = []
ye  = []
yf = []
yg  = []
yh = []
yi  = []
yj = []
slca = Slice()
slcb = Slice()
slcc = Slice()
slcd = Slice()
slce = Slice()
slcf = Slice()
slcg = Slice()
slch = Slice()
slci = Slice()
slcj = Slice()
aW1, aW2, aW3, aW4, aW5, ab1, ab2, ab3, ab4, ab5 = slca.getweightsDense4(na)
bW1, bW2, bW3, bW4, bW5, bb1, bb2, bb3, bb4, bb5 = slcb.getweightsDense4(nb)
cW1, cW2, cW3, cW4, cW5, cb1, cb2, cb3, cb4, cb5 = slcc.getweightsDense4(nc)
dW1, dW2, dW3, dW4, dW5, db1, db2, db3, db4, db5 = slcd.getweightsDense4(nd)
eW1, eW2, eW3, eW4, eW5, eb1, eb2, eb3, eb4, eb5 = slce.getweightsDense4(ne)
fW1, fW2, fW3, fW4, fW5, fb1, fb2, fb3, fb4, fb5 = slcf.getweightsDense4(nf)
gW1, gW2, gW3, gW4, gW5, gb1, gb2, gb3, gb4, gb5 = slcg.getweightsDense4(ng)
hW1, hW2, hW3, hW4, hW5, hb1, hb2, hb3, hb4, hb5 = slch.getweightsDense4(nh)
iW1, iW2, iW3, iW4, iW5, ib1, ib2, ib3, ib4, ib5 = slci.getweightsDense4(ni)
jW1, jW2, jW3, jW4, jW5, jb1, jb2, jb3, jb4, jb5 = slcj.getweightsDense4(nj)
la = 0
lb = 1
lc = 2
ld = 3
le = 4
lf = 5
lg = 6
lh = 7
li = 8
lj = 9
th1 = 0
th2 = 0
for i in range(0,len(yt)):
    if yt[i] == la and na.predict(xt[i:i+1])[0][la] >.9:
        ya.append(xt[i])
    if yt[i] == lb and nb.predict(xt[i:i+1])[0][lb] >.9:
        yb.append(xt[i])
    if yt[i] == lc and nc.predict(xt[i:i+1])[0][lc] >.9:
        yc.append(xt[i])
    if yt[i] == ld and nd.predict(xt[i:i+1])[0][ld] >.9:
        yd.append(xt[i])
    if yt[i] == le and ne.predict(xt[i:i+1])[0][le] >.9:
        ye.append(xt[i])
    if yt[i] == lf and nf.predict(xt[i:i+1])[0][lf] >.9:
        yf.append(xt[i])
    if yt[i] == lg and ng.predict(xt[i:i+1])[0][lg] >.9:
        yg.append(xt[i])
    if yt[i] == lh and nh.predict(xt[i:i+1])[0][lh] >.9:
        yh.append(xt[i])
    if yt[i] == li and ni.predict(xt[i:i+1])[0][li] >.9:
        yi.append(xt[i])
    if yt[i] == lj and nj.predict(xt[i:i+1])[0][lj] >.9:
        yj.append(xt[i])
print(len(ya),len(yb),len(yc),len(yd),len(ye),len(yf),len(yg),len(yh),len(yi),len(yj))

np.random.shuffle(ya)
np.random.shuffle(yb)
np.random.shuffle(yc)
np.random.shuffle(yd)
np.random.shuffle(ye)
np.random.shuffle(yf)
np.random.shuffle(yg)
np.random.shuffle(yh)
np.random.shuffle(yi)
np.random.shuffle(yj)
#ya = ya[0:100]
#yb = yb[0:100]
#yc = yc[0:100]
#yd = yd[0:100]
#ye = ye[0:100]
#yf = yf[0:100]
#yg = yg[0:100]
#yh = yh[0:100]
#yi = yi[0:100]
#yj = yj[0:100]
#slc.D1
sliceacc=[]
ac = 0
for x in ya:
    #W1, W2,b1,b2 = slc.dynamicmodify(nm,x,sx,sy)
    aW1, aW2, aW3, aW4, aW5, ab1, ab2, ab3, ab4, ab5 = slca.modifyThroughInterSectionDense4(na,x,sx,sy, th1)
    ac = ac + 1
    #print(np.count_nonzero(slcm.D2))
    if np.count_nonzero(slca.D5) < 45:
        print("Breaking at ac ", ac,np.count_nonzero(slca.D5))
        slca.first = True

bc = 0
for x in yb:
    #W1, W2,b1,b2 = slc.dynamicmodify(nm,x,sx,sy)
    bW1, bW2, bW3, bW4, bW5, bb1, bb2, bb3, bb4, bb5 = slcb.modifyThroughInterSectionDense4(nb,x,sx,sy, th2)
    bc = bc + 1
    if np.count_nonzero(slcb.D5) < 45:
        print("Breaking at bc ", bc,np.count_nonzero(slcb.D5))
        slcb.first = True
cc = 0
for x in yc:
    #W1, W2,b1,b2 = slc.dynamicmodify(nm,x,sx,sy)
    cW1, cW2, cW3, cW4, cW5, cb1, cb2, cb3, cb4, cb5 = slcc.modifyThroughInterSectionDense4(nc,x,sx,sy, th1)
    cc = cc + 1
    #print(np.count_nonzero(slcm.D2))
    if np.count_nonzero(slcc.D5) < 45:
        print("Breaking at cc ", cc,np.count_nonzero(slcc.D5))
        slcc.first = True

dc = 0
for x in yd:
    #W1, W2,b1,b2 = slc.dynamicmodify(nm,x,sx,sy)
    dW1, dW2, dW3, dW4, dW5, db1, db2, db3, db4, db5 = slcd.modifyThroughInterSectionDense4(nd,x,sx,sy, th2)
    dc = dc + 1
    if np.count_nonzero(slcd.D5) < 45:
        print("Breaking at dc ", dc,np.count_nonzero(slcd.D5))
        slcd.first = True
        
ec = 0
for x in ye:
    #W1, W2,b1,b2 = slc.dynamicmodify(nm,x,sx,sy)
    eW1, eW2, eW3, eW4, eW5, eb1, eb2, eb3, eb4, eb5 = slce.modifyThroughInterSectionDense4(ne,x,sx,sy, th1)
    ec = ec + 1
    #print(np.count_nonzero(slcm.D2))
    if np.count_nonzero(slce.D5) < 45:
        print("Breaking at ec ", ec,np.count_nonzero(slce.D5))
        slce.first = True

fc = 0
for x in yf:
    #W1, W2,b1,b2 = slc.dynamicmodify(nm,x,sx,sy)
    fW1, fW2, fW3, fW4, fW5, fb1, fb2, fb3, fb4, fb5 = slcf.modifyThroughInterSectionDense4(nf,x,sx,sy, th2)
    fc = fc + 1
    if np.count_nonzero(slcf.D5) < 45:
        print("Breaking at fc ", fc,np.count_nonzero(slcf.D5))
        slcf.first = True
        
gc = 0
for x in yg:
    #W1, W2,b1,b2 = slc.dynamicmodify(nm,x,sx,sy)
    gW1, gW2, gW3, gW4, gW5, gb1, gb2, gb3, gb4, gb5 = slcg.modifyThroughInterSectionDense4(ng,x,sx,sy, th1)
    gc = gc + 1
    #print(np.count_nonzero(slcm.D2))
    if np.count_nonzero(slcg.D5) < 45:
        print("Breaking at gc ", gc,np.count_nonzero(slcg.D5))
        slcg.first = True

hc = 0
for x in yh:
    #W1, W2,b1,b2 = slc.dynamicmodify(nm,x,sx,sy)
    hW1, hW2, hW3, hW4, hW5, hb1, hb2, hb3, hb4, hb5 = slch.modifyThroughInterSectionDense4(nh,x,sx,sy, th2)
    hc = hc + 1
    if np.count_nonzero(slch.D5) < 45:
        print("Breaking at hc ", hc,np.count_nonzero(slch.D5))
        slch.first = True
ic = 0
for x in yi:
    #W1, W2,b1,b2 = slc.dynamicmodify(nm,x,sx,sy)
    iW1, iW2, iW3, iW4, iW5, ib1, ib2, ib3, ib4, ib5 = slci.modifyThroughInterSectionDense4(ni,x,sx,sy, th1)
    ic = ic + 1
    #print(np.count_nonzero(slcm.D2))
    if np.count_nonzero(slci.D5) < 45:
        print("Breaking at ic ", ic,np.count_nonzero(slci.D5))
        slci.first = True

jc = 0
for x in yj:
    #W1, W2,b1,b2 = slc.dynamicmodify(nm,x,sx,sy)
    jW1, jW2, jW3, jW4, jW5, jb1, jb2, jb3, jb4, jb5 = slcj.modifyThroughInterSectionDense4(nj,x,sx,sy, th2)
    jc = jc + 1
    if np.count_nonzero(slcj.D5) < 45:
        print("Breaking at jc ", jc,np.count_nonzero(slcj.D5))
        slcj.first = True


# In[ ]:


#0
na.layers[1].set_weights([slca.D1,slca.d1])
na.layers[2].set_weights([slca.D2,slca.d2])
na.layers[3].set_weights([slca.D3,slca.d3])
na.layers[4].set_weights([slca.D4,slca.d4])
na.layers[5].set_weights([slca.D5,slca.d5])
#print(model.get_weights())
from sklearn.metrics import accuracy_score
zeros = []
pred = []
tr = []
labs = [0,1,2,3,4,5,6,7,8,9]
acc = []
count = 0
for ly in labs:
    pred = []
    tr = []
    for i in range(0,len(yt)):
        count += 1
        if yt[i] == ly:
            p = na.predict(xt[i:i+1])
            m = p.argmax()
            pred.append(m)
            tr.append(ly)
    score = accuracy_score(pred,tr)
    acc.append(score)
print(acc)
print(count)
sliceacc.append(acc)

# In[ ]:


#1
nb.layers[1].set_weights([slcb.D1,slcb.d1])
nb.layers[2].set_weights([slcb.D2,slcb.d2])
nb.layers[3].set_weights([slcb.D3,slcb.d3])
nb.layers[4].set_weights([slcb.D4,slcb.d4])
nb.layers[5].set_weights([slcb.D5,slcb.d5])
#print(model.get_weights())
from sklearn.metrics import accuracy_score
zeros = []
pred = []
tr = []
labs = [0,1,2,3,4,5,6,7,8,9]
acc = []
count = 0
for ly in labs:
    pred = []
    tr = []
    for i in range(0,len(yt)):
        count += 1
        if yt[i] == ly:
            p = nb.predict(xt[i:i+1])
            m = p.argmax()
            pred.append(m)
            tr.append(ly)
    score = accuracy_score(pred,tr)
    acc.append(score)
print(acc)
print(count)
sliceacc.append(acc)

# In[ ]:


#2
nc.layers[1].set_weights([slcc.D1,slcc.d1])
nc.layers[2].set_weights([slcc.D2,slcc.d2])
nc.layers[3].set_weights([slcc.D3,slcc.d3])
nc.layers[4].set_weights([slcc.D4,slcc.d4])
nc.layers[5].set_weights([slcc.D5,slcc.d5])
#print(model.get_weights())
from sklearn.metrics import accuracy_score
zeros = []
pred = []
tr = []
labs = [0,1,2,3,4,5,6,7,8,9]
acc = []
count = 0
for ly in labs:
    pred = []
    tr = []
    for i in range(0,len(yt)):
        count += 1
        if yt[i] == ly:
            p = nc.predict(xt[i:i+1])
            m = p.argmax()
            pred.append(m)
            tr.append(ly)
    score = accuracy_score(pred,tr)
    acc.append(score)
print(acc)
print(count)

sliceacc.append(acc)
# In[ ]:


#3
nd.layers[1].set_weights([slcd.D1,slcd.d1])
nd.layers[2].set_weights([slcd.D2,slcd.d2])
nd.layers[3].set_weights([slcd.D3,slcd.d3])
nd.layers[4].set_weights([slcd.D4,slcd.d4])
nd.layers[5].set_weights([slcd.D5,slcd.d5])
#print(model.get_weights())
from sklearn.metrics import accuracy_score
zeros = []
pred = []
tr = []
labs = [0,1,2,3,4,5,6,7,8,9]
acc = []
count = 0
for ly in labs:
    pred = []
    tr = []
    for i in range(0,len(yt)):
        count += 1
        if yt[i] == ly:
            p = nd.predict(xt[i:i+1])
            m = p.argmax()
            pred.append(m)
            tr.append(ly)
    score = accuracy_score(pred,tr)
    acc.append(score)
print(acc)
print(count)

sliceacc.append(acc)
# In[ ]:


#4
ne.layers[1].set_weights([slce.D1,slce.d1])
ne.layers[2].set_weights([slce.D2,slce.d2])
ne.layers[3].set_weights([slce.D3,slce.d3])
ne.layers[4].set_weights([slce.D4,slce.d4])
ne.layers[5].set_weights([slce.D5,slce.d5])
#print(model.get_weights())
from sklearn.metrics import accuracy_score
zeros = []
pred = []
tr = []
labs = [0,1,2,3,4,5,6,7,8,9]
acc = []
count = 0
for ly in labs:
    pred = []
    tr = []
    for i in range(0,len(yt)):
        count += 1
        if yt[i] == ly:
            p = ne.predict(xt[i:i+1])
            m = p.argmax()
            pred.append(m)
            tr.append(ly)
    score = accuracy_score(pred,tr)
    acc.append(score)
print(acc)
print(count)

sliceacc.append(acc)
# In[ ]:


#5
nf.layers[1].set_weights([slcf.D1,slcf.d1])
nf.layers[2].set_weights([slcf.D2,slcf.d2])
nf.layers[3].set_weights([slcf.D3,slcf.d3])
nf.layers[4].set_weights([slcf.D4,slcf.d4])
nf.layers[5].set_weights([slcf.D5,slcf.d5])
#print(model.get_weights())
from sklearn.metrics import accuracy_score
zeros = []
pred = []
tr = []
labs = [0,1,2,3,4,5,6,7,8,9]
acc = []
count = 0
for ly in labs:
    pred = []
    tr = []
    for i in range(0,len(yt)):
        count += 1
        if yt[i] == ly:
            p = nf.predict(xt[i:i+1])
            m = p.argmax()
            pred.append(m)
            tr.append(ly)
    score = accuracy_score(pred,tr)
    acc.append(score)
print(acc)
print(count)
sliceacc.append(acc)

# In[ ]:


#6
ng.layers[1].set_weights([slcg.D1,slcg.d1])
ng.layers[2].set_weights([slcg.D2,slcg.d2])
ng.layers[3].set_weights([slcg.D3,slcg.d3])
ng.layers[4].set_weights([slcg.D4,slcg.d4])
ng.layers[5].set_weights([slcg.D5,slcg.d5])
#print(model.get_weights())
from sklearn.metrics import accuracy_score
zeros = []
pred = []
tr = []
labs = [0,1,2,3,4,5,6,7,8,9]
acc = []
count = 0
for ly in labs:
    pred = []
    tr = []
    for i in range(0,len(yt)):
        count += 1
        if yt[i] == ly:
            p = ng.predict(xt[i:i+1])
            m = p.argmax()
            pred.append(m)
            tr.append(ly)
    score = accuracy_score(pred,tr)
    acc.append(score)
print(acc)
print(count)

sliceacc.append(acc)
# In[ ]:


#7
nh.layers[1].set_weights([slch.D1,slch.d1])
nh.layers[2].set_weights([slch.D2,slch.d2])
nh.layers[3].set_weights([slch.D3,slch.d3])
nh.layers[4].set_weights([slch.D4,slch.d4])
nh.layers[5].set_weights([slch.D5,slch.d5])
#print(model.get_weights())
from sklearn.metrics import accuracy_score
zeros = []
pred = []
tr = []
labs = [0,1,2,3,4,5,6,7,8,9]
acc = []
count = 0
for ly in labs:
    pred = []
    tr = []
    for i in range(0,len(yt)):
        count += 1
        if yt[i] == ly:
            p = nh.predict(xt[i:i+1])
            m = p.argmax()
            pred.append(m)
            tr.append(ly)
    score = accuracy_score(pred,tr)
    acc.append(score)
print(acc)
print(count)

sliceacc.append(acc)
# In[ ]:


#8
ni.layers[1].set_weights([slci.D1,slci.d1])
ni.layers[2].set_weights([slci.D2,slci.d2])
ni.layers[3].set_weights([slci.D3,slci.d3])
ni.layers[4].set_weights([slci.D4,slci.d4])
ni.layers[5].set_weights([slci.D5,slci.d5])
#print(model.get_weights())
from sklearn.metrics import accuracy_score
zeros = []
pred = []
tr = []
labs = [0,1,2,3,4,5,6,7,8,9]
acc = []
count = 0
for ly in labs:
    pred = []
    tr = []
    for i in range(0,len(yt)):
        count += 1
        if yt[i] == ly:
            p = ni.predict(xt[i:i+1])
            m = p.argmax()
            pred.append(m)
            tr.append(ly)
    score = accuracy_score(pred,tr)
    acc.append(score)
print(acc)
print(count)

sliceacc.append(acc)
# In[ ]:


#9
nj.layers[1].set_weights([slcj.D1,slcj.d1])
nj.layers[2].set_weights([slcj.D2,slcj.d2])
nj.layers[3].set_weights([slcj.D3,slcj.d3])
nj.layers[4].set_weights([slcj.D4,slcj.d4])
nj.layers[5].set_weights([slcj.D5,slcj.d5])
#print(model.get_weights())
from sklearn.metrics import accuracy_score
zeros = []
pred = []
tr = []
labs = [0,1,2,3,4,5,6,7,8,9]
acc = []
count = 0
for ly in labs:
    pred = []
    tr = []
    for i in range(0,len(yt)):
        count += 1
        if yt[i] == ly:
            p = nj.predict(xt[i:i+1])
            m = p.argmax()
            pred.append(m)
            tr.append(ly)
    score = accuracy_score(pred,tr)
    acc.append(score)
print(acc)
print(count)

sliceacc.append(acc)
# In[ ]:


sliceacc1=sliceacc
sliceacc=np.array(sliceacc)


# In[ ]:

zeroLayer1=np.nonzero(aW1[0])
OneLayer1=np.nonzero(bW1[0])
TwoLayer1=np.nonzero(cW1[0])
ThreeLayer1=np.nonzero(dW1[0])
FourLayer1=np.nonzero(eW1[0])
FiveLayer1=np.nonzero(fW1[0])
SixLayer1=np.nonzero(gW1[0])
SevenLayer1=np.nonzero(hW1[0])
EightLayer1=np.nonzero(iW1[0])
NineLayer1=np.nonzero(jW1[0])


zeroLayer1=np.array(zeroLayer1)
OneLayer1=np.array(OneLayer1)
TwoLayer1=np.array(TwoLayer1)
ThreeLayer1=np.array(ThreeLayer1)
FourLayer1=np.array(FourLayer1)
FiveLayer1=np.array(FiveLayer1)
SixLayer1=np.array(SixLayer1)
SevenLayer1=np.array(SevenLayer1)
EightLayer1=np.array(EightLayer1)
NineLayer1=np.array(NineLayer1)




zeroLayer2=np.nonzero(aW2[0])
OneLayer2=np.nonzero(bW2[0])
TwoLayer2=np.nonzero(cW2[0])
ThreeLayer2=np.nonzero(dW2[0])
FourLayer2=np.nonzero(eW2[0])
FiveLayer2=np.nonzero(fW2[0])
SixLayer2=np.nonzero(gW2[0])
SevenLayer2=np.nonzero(hW2[0])
EightLayer2=np.nonzero(iW2[0])
NineLayer2=np.nonzero(jW2[0])



zeroLayer2=np.array(zeroLayer2)
OneLayer2=np.array(OneLayer2)
TwoLayer2=np.array(TwoLayer2)
ThreeLayer2=np.array(ThreeLayer2)
FourLayer2=np.array(FourLayer2)
FiveLayer2=np.array(FiveLayer2)
SixLayer2=np.array(SixLayer2)
SevenLayer2=np.array(SevenLayer2)
EightLayer2=np.array(EightLayer2)
NineLayer2=np.array(NineLayer2)

zeroLayer3=np.nonzero(aW3[0])
OneLayer3=np.nonzero(bW3[0])
TwoLayer3=np.nonzero(cW3[0])
ThreeLayer3=np.nonzero(dW3[0])
FourLayer3=np.nonzero(eW3[0])
FiveLayer3=np.nonzero(fW3[0])
SixLayer3=np.nonzero(gW3[0])
SevenLayer3=np.nonzero(hW3[0])
EightLayer3=np.nonzero(iW3[0])
NineLayer3=np.nonzero(jW3[0])


zeroLayer3=np.array(zeroLayer3)
OneLayer3=np.array(OneLayer3)
TwoLayer3=np.array(TwoLayer3)
ThreeLayer3=np.array(ThreeLayer3)
FourLayer3=np.array(FourLayer3)
FiveLayer3=np.array(FiveLayer3)
SixLayer3=np.array(SixLayer3)
SevenLayer3=np.array(SevenLayer3)
EightLayer3=np.array(EightLayer3)
NineLayer3=np.array(NineLayer3)




zeroLayer4=np.nonzero(aW4[0])
OneLayer4=np.nonzero(bW4[0])
TwoLayer4=np.nonzero(cW4[0])
ThreeLayer4=np.nonzero(dW4[0])
FourLayer4=np.nonzero(eW4[0])
FiveLayer4=np.nonzero(fW4[0])
SixLayer4=np.nonzero(gW4[0])
SevenLayer4=np.nonzero(hW4[0])
EightLayer4=np.nonzero(iW4[0])
NineLayer4=np.nonzero(jW4[0])



zeroLayer4=np.array(zeroLayer4)
OneLayer4=np.array(OneLayer4)
TwoLayer4=np.array(TwoLayer4)
ThreeLayer4=np.array(ThreeLayer4)
FourLayer4=np.array(FourLayer4)
FiveLayer4=np.array(FiveLayer4)
SixLayer4=np.array(SixLayer4)
SevenLayer4=np.array(SevenLayer4)
EightLayer4=np.array(EightLayer4)
NineLayer4=np.array(NineLayer4)


# In[ ]:
#Scoring for training dataset:
testSlice=Slice()
testW1, testW2, testW3, testW4, testW5, testb1, testb2, testb3, testb4, testb5 = testSlice.getweightsDense4(nj)
xc=0
W1list=[]
W2list=[]
W3list=[]
W4list=[]
for x in xt:
    #W1, W2,b1,b2 = slc.dynamicmodify(nm,x,sx,sy)
    testSlice=Slice()
    testW1, testW2, testW3, testW4, testW5, testb1, testb2, testb3, testb4, testb5 = testSlice.getweightsDense4(nj)
    testW1, testW2, testW3, testW4, testW5, testb1, testb2, testb3, testb4, testb5 = testSlice.modifyThroughInterSectionDense4(nj,x,sx,sy, th2)
    W1list.append(np.array(np.nonzero(testW1[0])))
    W2list.append(np.array(np.nonzero(testW2[0])))
    W3list.append(np.array(np.nonzero(testW3[0])))
    W4list.append(np.array(np.nonzero(testW4[0])))
    xc = xc + 1
    if np.count_nonzero(testSlice.D5) < 45:
        print("Breaking at xc ", xc,np.count_nonzero(testSlice.D5))
        testSlice.first = True

# In[ ]:
# Calculate the hamming distance
def hamming_distance(string1, string2): 
    distance = 0
    L = len(string1)
    for i in range(L):
        if string1[i] != string2[i]:
            distance += 1
    return distance

# In[]:
# Jaccard Similarity:
def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    print(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection / union)

# In[]: Dice Similarity
from scipy.spatial import distance
#distance.dice(list1, list 2)
# In[]:
#Cosine Similarity
from sklearn.metrics.pairwise import cosine_similarity
#cosine_similarity(list1, list2)
# In[]:
# Tanimoto Similarity

# In[]:
# Apply Hamming Distance
# S0L1
S0L1=""
for i in range(0,49):
    S0L1+="0"
S0L2=S0L1
S0L3=S0L1
S0L4=S0L1

S1L1=S0L1
S1L2=S0L1
S1L3=S0L1
S1L4=S0L1

S2L1=S0L1
S2L2=S0L1
S2L3=S0L1
S2L4=S0L1

S3L1=S0L1
S3L2=S0L1
S3L3=S0L1
S3L4=S0L1

S4L1=S0L1
S4L2=S0L1
S4L3=S0L1
S4L4=S0L1

S5L1=S0L1
S5L2=S0L1
S5L3=S0L1
S5L4=S0L1

S6L1=S0L1
S6L2=S0L1
S6L3=S0L1
S6L4=S0L1

S7L1=S0L1
S7L2=S0L1
S7L3=S0L1
S7L4=S0L1

S8L1=S0L1
S8L2=S0L1
S8L3=S0L1
S8L4=S0L1

S9L1=S0L1
S9L2=S0L1
S9L3=S0L1
S9L4=S0L1

#for i in range(0,49):
#    if np.any(zeroLayer1[:, 0] == i)==True:
#        S0L1[i]="1"
# In[]:
#S0L1
for i in range(0,49):
    S0L1list=list(S0L1)
    if np.any(zeroLayer1[0, :] == i)==True:
        S0L1list[i]="1"
        S0L1="".join(S0L1list)
# In[]:
#S0L2
for i in range(0,49):
    S0L2list=list(S0L2)
    if np.any(zeroLayer2[0, :] == i)==True:
        S0L2list[i]="1"
        S0L2="".join(S0L2list)
# In[]:
#S0L3
for i in range(0,49):
    S0L3list=list(S0L3)
    if np.any(zeroLayer3[0, :] == i)==True:
        S0L3list[i]="1"
        S0L3="".join(S0L3list)
# In[]:
#S0L4
for i in range(0,49):
    S0L4list=list(S0L4)
    if np.any(zeroLayer4[0, :] == i)==True:
        S0L4list[i]="1"
        S0L4="".join(S0L4list)
# In[]:
#S1L1
for i in range(0,49):
    S1L1list=list(S1L1)
    if np.any(OneLayer1[0, :] == i)==True:
        S1L1list[i]="1"
        S1L1="".join(S1L1list)
# In[]:
#S1L2
for i in range(0,49):
    S1L2list=list(S1L2)
    if np.any(OneLayer2[0, :] == i)==True:
        S1L2list[i]="1"
        S1L2="".join(S1L2list)
# In[]:
#S1L3
for i in range(0,49):
    S1L3list=list(S1L3)
    if np.any(OneLayer3[0, :] == i)==True:
        S1L3list[i]="1"
        S1L3="".join(S1L3list)
# In[]:
#S1L4
for i in range(0,49):
    S1L4list=list(S1L4)
    if np.any(OneLayer4[0, :] == i)==True:
        S1L4list[i]="1"
        S1L4="".join(S1L4list)
# In[]:
#S2L1
for i in range(0,49):
    S2L1list=list(S2L1)
    if np.any(TwoLayer1[0, :] == i)==True:
        S2L1list[i]="1"
        S2L1="".join(S2L1list)
# In[]:
#S2L2
for i in range(0,49):
    S2L2list=list(S2L2)
    if np.any(TwoLayer2[0, :] == i)==True:
        S2L2list[i]="1"
        S2L2="".join(S2L2list)
# In[]:
#S2L3
for i in range(0,49):
    S2L3list=list(S2L3)
    if np.any(TwoLayer3[0, :] == i)==True:
        S2L3list[i]="1"
        S2L3="".join(S2L3list)
# In[]:
#S2L4
for i in range(0,49):
    S2L4list=list(S2L4)
    if np.any(TwoLayer4[0, :] == i)==True:
        S2L4list[i]="1"
        S2L4="".join(S2L4list)
# In[]:
#S3L1
for i in range(0,49):
    S3L1list=list(S3L1)
    if np.any(ThreeLayer1[0, :] == i)==True:
        S3L1list[i]="1"
        S3L1="".join(S3L1list)
# In[]:
#S3L2
for i in range(0,49):
    S3L2list=list(S3L2)
    if np.any(ThreeLayer2[0, :] == i)==True:
        S3L2list[i]="1"
        S3L2="".join(S3L2list)
# In[]:
#S3L3
for i in range(0,49):
    S3L3list=list(S3L3)
    if np.any(ThreeLayer3[0, :] == i)==True:
        S3L3list[i]="1"
        S3L3="".join(S3L3list)
# In[]:
#S3L4
for i in range(0,49):
    S3L4list=list(S3L4)
    if np.any(ThreeLayer4[0, :] == i)==True:
        S3L4list[i]="1"
        S3L4="".join(S3L4list)
# In[]:
#S4L1
for i in range(0,49):
    S4L1list=list(S4L1)
    if np.any(FourLayer1[0, :] == i)==True:
        S4L1list[i]="1"
        S4L1="".join(S4L1list)
# In[]:
#S4L2
for i in range(0,49):
    S4L2list=list(S4L2)
    if np.any(FourLayer2[0, :] == i)==True:
        S4L2list[i]="1"
        S4L2="".join(S4L2list)
# In[]:
#S4L3
for i in range(0,49):
    S4L3list=list(S4L3)
    if np.any(FourLayer3[0, :] == i)==True:
        S4L3list[i]="1"
        S4L3="".join(S4L3list)
# In[]:
#S4L4
for i in range(0,49):
    S4L4list=list(S4L4)
    if np.any(FourLayer4[0, :] == i)==True:
        S4L4list[i]="1"
        S4L4="".join(S4L4list)
# In[]:
#S5L1
for i in range(0,49):
    S5L1list=list(S5L1)
    if np.any(FiveLayer1[0, :] == i)==True:
        S5L1list[i]="1"
        S5L1="".join(S5L1list)
# In[]:
#S5L2
for i in range(0,49):
    S5L2list=list(S5L2)
    if np.any(FiveLayer2[0, :] == i)==True:
        S5L2list[i]="1"
        S5L2="".join(S5L2list)
# In[]:
#S5L3
for i in range(0,49):
    S5L3list=list(S5L3)
    if np.any(FiveLayer3[0, :] == i)==True:
        S5L3list[i]="1"
        S5L3="".join(S5L3list)
# In[]:
#S5L4
for i in range(0,49):
    S5L4list=list(S5L4)
    if np.any(FiveLayer4[0, :] == i)==True:
        S5L4list[i]="1"
        S5L4="".join(S5L4list)
# In[]:
#S6L1
for i in range(0,49):
    S6L1list=list(S6L1)
    if np.any(SixLayer1[0, :] == i)==True:
        S6L1list[i]="1"
        S6L1="".join(S6L1list)
# In[]:
#S6L2
for i in range(0,49):
    S6L2list=list(S6L2)
    if np.any(SixLayer2[0, :] == i)==True:
        S6L2list[i]="1"
        S6L2="".join(S6L2list)
# In[]:
#S6L3
for i in range(0,49):
    S6L3list=list(S6L3)
    if np.any(SixLayer3[0, :] == i)==True:
        S6L3list[i]="1"
        S6L3="".join(S6L3list)
# In[]:
#S6L4
for i in range(0,49):
    S6L4list=list(S6L4)
    if np.any(SixLayer4[0, :] == i)==True:
        S6L4list[i]="1"
        S6L4="".join(S6L4list)
# In[]:
#S7L1
for i in range(0,49):
    S7L1list=list(S7L1)
    if np.any(SevenLayer1[0, :] == i)==True:
        S7L1list[i]="1"
        S7L1="".join(S7L1list)
# In[]:
#S7L2
for i in range(0,49):
    S7L2list=list(S7L2)
    if np.any(SevenLayer2[0, :] == i)==True:
        S7L2list[i]="1"
        S7L2="".join(S7L2list)
# In[]:
#S7L3
for i in range(0,49):
    S7L3list=list(S7L3)
    if np.any(SevenLayer3[0, :] == i)==True:
        S7L3list[i]="1"
        S7L3="".join(S7L3list)
# In[]:
#S7L4
for i in range(0,49):
    S7L4list=list(S7L4)
    if np.any(SevenLayer4[0, :] == i)==True:
        S7L4list[i]="1"
        S7L4="".join(S7L4list)
# In[]:
#S8L1
for i in range(0,49):
    S8L1list=list(S8L1)
    if np.any(EightLayer1[0, :] == i)==True:
        S0L1list[i]="1"
        S8L1="".join(S8L1list)
# In[]:
#S8L2
for i in range(0,49):
    S8L2list=list(S8L2)
    if np.any(EightLayer2[0, :] == i)==True:
        S8L2list[i]="1"
        S8L2="".join(S8L2list)
# In[]:
#S8L3
for i in range(0,49):
    S8L3list=list(S8L3)
    if np.any(EightLayer3[0, :] == i)==True:
        S8L3list[i]="1"
        S8L3="".join(S8L3list)
# In[]:
#S8L4
for i in range(0,49):
    S8L4list=list(S8L4)
    if np.any(EightLayer4[0, :] == i)==True:
        S8L4list[i]="1"
        S8L4="".join(S8L4list)
# In[]:
#S9L1
for i in range(0,49):
    S9L1list=list(S9L1)
    if np.any(NineLayer1[0, :] == i)==True:
        S9L1list[i]="1"
        S9L1="".join(S9L1list)
# In[]:
#S9L2
for i in range(0,49):
    S9L2list=list(S9L2)
    if np.any(NineLayer2[0, :] == i)==True:
        S9L2list[i]="1"
        S9L2="".join(S9L2list)
# In[]:
#S9L3
for i in range(0,49):
    S9L3list=list(S9L3)
    if np.any(NineLayer3[0, :] == i)==True:
        S9L3list[i]="1"
        S9L3="".join(S9L3list)
# In[]:
#S9L4
for i in range(0,49):
    S9L4list=list(S9L4)
    if np.any(NineLayer4[0, :] == i)==True:
        S9L4list[i]="1"
        S9L4="".join(S9L4list)

# In[]: Node to Binary L1
W1listBinary=[]
for x in range(0,60000):
    W1listStr=""
    for i in range(0,49):
        W1listStr+="0"
    for y in range(0,49):
        W1listStrlist=list(W1listStr)
        if np.any((W1list[x])[0, :] == y)==True:
            W1listStrlist[y]="1"
            W1listStr="".join(W1listStrlist)
    W1listBinary.append(W1listStr)
# In[]:Node to Binary L2
W2listBinary=[]
for x in range(0,60000):
    W2listStr=""
    for i in range(0,49):
        W2listStr+="0"
    for y in range(0,49):
        W2listStrlist=list(W2listStr)
        if np.any((W2list[x])[0, :] == y)==True:
            W2listStrlist[y]="1"
            W2listStr="".join(W2listStrlist)
    W2listBinary.append(W2listStr)
# In[]:Node to Binary L3
W3listBinary=[]
for x in range(0,60000):
    W3listStr=""
    for i in range(0,49):
        W3listStr+="0"
    for y in range(0,49):
        W3listStrlist=list(W3listStr)
        if np.any((W3list[x])[0, :] == y)==True:
            W3listStrlist[y]="1"
            W3listStr="".join(W3listStrlist)
    W3listBinary.append(W3listStr)
# In[]:Node to Binary L4
W4listBinary=[]
for x in range(0,60000):
    W4listStr=""
    for i in range(0,49):
        W4listStr+="0"
    for y in range(0,49):
        W4listStrlist=list(W4listStr)
        if np.any((W4list[x])[0, :] == y)==True:
            W4listStrlist[y]="1"
            W4listStr="".join(W4listStrlist)
    W4listBinary.append(W4listStr) 
# In[]:
#Calculate Hamming Distance L1
W1HammingDistance=[]
for x in W1listBinary:
    temp=[]
    S0L1Distance=hamming_distance(S0L1, x)
    S1L1Distance=hamming_distance(S1L1, x)
    S2L1Distance=hamming_distance(S2L1, x)
    S3L1Distance=hamming_distance(S3L1, x)
    S4L1Distance=hamming_distance(S4L1, x)
    S5L1Distance=hamming_distance(S5L1, x)
    S6L1Distance=hamming_distance(S6L1, x)
    S7L1Distance=hamming_distance(S7L1, x)
    S8L1Distance=hamming_distance(S8L1, x)
    S9L1Distance=hamming_distance(S9L1, x)
    temp.append(S0L1Distance)
    temp.append(S1L1Distance)
    temp.append(S2L1Distance)
    temp.append(S3L1Distance)
    temp.append(S4L1Distance)
    temp.append(S5L1Distance)
    temp.append(S6L1Distance)
    temp.append(S7L1Distance)
    temp.append(S8L1Distance)
    temp.append(S9L1Distance)
    W1HammingDistance.append(temp)
# In[]:
#Calculate Hamming Distance L2
W2HammingDistance=[]
for x in W2listBinary:
    temp=[]
    S0L2Distance=hamming_distance(S0L2, x)
    S1L2Distance=hamming_distance(S1L2, x)
    S2L2Distance=hamming_distance(S2L2, x)
    S3L2Distance=hamming_distance(S3L2, x)
    S4L2Distance=hamming_distance(S4L2, x)
    S5L2Distance=hamming_distance(S5L2, x)
    S6L2Distance=hamming_distance(S6L2, x)
    S7L2Distance=hamming_distance(S7L2, x)
    S8L2Distance=hamming_distance(S8L2, x)
    S9L2Distance=hamming_distance(S9L2, x)
    temp.append(S0L2Distance)
    temp.append(S1L2Distance)
    temp.append(S2L2Distance)
    temp.append(S3L2Distance)
    temp.append(S4L2Distance)
    temp.append(S5L2Distance)
    temp.append(S6L2Distance)
    temp.append(S7L2Distance)
    temp.append(S8L2Distance)
    temp.append(S9L2Distance)
    W2HammingDistance.append(temp)
    # In[]:
#Calculate Hamming Distance L3
W3HammingDistance=[]
for x in W1listBinary:
    temp=[]
    S0L3Distance=hamming_distance(S0L3, x)
    S1L3Distance=hamming_distance(S1L3, x)
    S2L3Distance=hamming_distance(S2L3, x)
    S3L3Distance=hamming_distance(S3L3, x)
    S4L3Distance=hamming_distance(S4L3, x)
    S5L3Distance=hamming_distance(S5L3, x)
    S6L3Distance=hamming_distance(S6L3, x)
    S7L3Distance=hamming_distance(S7L3, x)
    S8L3Distance=hamming_distance(S8L3, x)
    S9L3Distance=hamming_distance(S9L3, x)
    temp.append(S0L3Distance)
    temp.append(S1L3Distance)
    temp.append(S2L3Distance)
    temp.append(S3L3Distance)
    temp.append(S4L3Distance)
    temp.append(S5L3Distance)
    temp.append(S6L3Distance)
    temp.append(S7L3Distance)
    temp.append(S8L3Distance)
    temp.append(S9L3Distance)
    W3HammingDistance.append(temp)
    # In[]:
#Calculate Hamming Distance L4
W4HammingDistance=[]
for x in W1listBinary:
    temp=[]
    S0L4Distance=hamming_distance(S0L4, x)
    S1L4Distance=hamming_distance(S1L4, x)
    S2L4Distance=hamming_distance(S2L4, x)
    S3L4Distance=hamming_distance(S3L4, x)
    S4L4Distance=hamming_distance(S4L4, x)
    S5L4Distance=hamming_distance(S5L4, x)
    S6L4Distance=hamming_distance(S6L4, x)
    S7L4Distance=hamming_distance(S7L4, x)
    S8L4Distance=hamming_distance(S8L4, x)
    S9L4Distance=hamming_distance(S9L4, x)
    temp.append(S0L4Distance)
    temp.append(S1L4Distance)
    temp.append(S2L4Distance)
    temp.append(S3L4Distance)
    temp.append(S4L4Distance)
    temp.append(S5L4Distance)
    temp.append(S6L4Distance)
    temp.append(S7L4Distance)
    temp.append(S8L4Distance)
    temp.append(S9L4Distance)
    W4HammingDistance.append(temp)
# In[]
hammingdistArr1=np.array(W1HammingDistance)
hammingdistArr2=np.array(W2HammingDistance)
hammingdistArr3=np.array(W3HammingDistance)
hammingdistArr4=np.array(W4HammingDistance)

# In[]
# Check predict
nj.layers[1].set_weights([testSlice.D1,testSlice.d1])
nj.layers[2].set_weights([testSlice.D2,testSlice.d2])
nj.layers[3].set_weights([testSlice.D3,testSlice.d3])
nj.layers[4].set_weights([testSlice.D4,testSlice.d4])
nj.layers[5].set_weights([testSlice.D5,testSlice.d5])
#print(model.get_weights())
from sklearn.metrics import accuracy_score
zeros = []
pred = []
tr = []
acc = []
count = 0
for i in range(0,len(xt)):
    p = nj.predict(xt[i:i+1])
    m = p.argmax()
    pred.append(m)
score = accuracy_score(pred,tr)
acc.append(score)
print(acc)
print(count)
