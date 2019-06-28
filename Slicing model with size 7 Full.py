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
na , xt, yt = mn.train2(X, Y, x,y,sx,sy,10,50)
nb , xt1, yt1 = na , xt, yt
nc , xt, yt = na , xt, yt
nd , xt1, yt1 = na , xt, yt
ne , xt, yt = na , xt, yt
nf , xt1, yt1 = na , xt, yt
ng , xt, yt = na , xt, yt
nh , xt1, yt1 = na , xt, yt
ni , xt, yt = na , xt, yt
nj , xt1, yt1 = na , xt, yt


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
aW1, aW2, ab1, ab2 = slca.getweights(na)
bW1, bW2, bb1, bb2 = slcb.getweights(nb)
cW1, cW2, cb1, cb2 = slcc.getweights(nc)
dW1, dW2, db1, db2 = slcd.getweights(nd)
eW1, eW2, eb1, eb2 = slce.getweights(ne)
fW1, fW2, fb1, fb2 = slcf.getweights(nf)
gW1, gW2, gb1, gb2 = slcg.getweights(ng)
hW1, hW2, hb1, hb2 = slch.getweights(nh)
iW1, iW2, ib1, ib2 = slci.getweights(ni)
jW1, jW2, ib1, jb2 = slcj.getweights(nj)
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
th1 = .0001
th2 = .0001
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
    aW1, aW2,ab1,ab2 = slca.modifyThroughInterSection(na,x,sx,sy, th1)
    ac = ac + 1
    #print(np.count_nonzero(slcm.D2))
    if np.count_nonzero(slca.D2) < 45:
        print("Breaking at ac ", ac,np.count_nonzero(slca.D2))
        slca.first = True

bc = 0
for x in yb:
    #W1, W2,b1,b2 = slc.dynamicmodify(nm,x,sx,sy)
    bW1, bW2,bb1,bb2 = slcb.modifyThroughInterSection(nb,x,sx,sy, th2)
    bc = bc + 1
    if np.count_nonzero(slcb.D2) < 45:
        print("Breaking at bc ", bc,np.count_nonzero(slcb.D2))
        slcb.first = True
cc = 0
for x in yc:
    #W1, W2,b1,b2 = slc.dynamicmodify(nm,x,sx,sy)
    cW1, cW2,cb1,cb2 = slcc.modifyThroughInterSection(nc,x,sx,sy, th1)
    cc = cc + 1
    #print(np.count_nonzero(slcm.D2))
    if np.count_nonzero(slcc.D2) < 45:
        print("Breaking at cc ", cc,np.count_nonzero(slcc.D2))
        slcc.first = True

dc = 0
for x in yd:
    #W1, W2,b1,b2 = slc.dynamicmodify(nm,x,sx,sy)
    dW1, dW2,db1,db2 = slcd.modifyThroughInterSection(nd,x,sx,sy, th2)
    dc = dc + 1
    if np.count_nonzero(slcd.D2) < 45:
        print("Breaking at dc ", dc,np.count_nonzero(slcd.D2))
        slcd.first = True
        
ec = 0
for x in ye:
    #W1, W2,b1,b2 = slc.dynamicmodify(nm,x,sx,sy)
    eW1, eW2,eb1,eb2 = slce.modifyThroughInterSection(ne,x,sx,sy, th1)
    ec = ec + 1
    #print(np.count_nonzero(slcm.D2))
    if np.count_nonzero(slce.D2) < 45:
        print("Breaking at ec ", ec,np.count_nonzero(slce.D2))
        slce.first = True

fc = 0
for x in yf:
    #W1, W2,b1,b2 = slc.dynamicmodify(nm,x,sx,sy)
    fW1, fW2,fb1,fb2 = slcf.modifyThroughInterSection(nf,x,sx,sy, th2)
    fc = fc + 1
    if np.count_nonzero(slcf.D2) < 45:
        print("Breaking at fc ", fc,np.count_nonzero(slcf.D2))
        slcf.first = True
        
gc = 0
for x in yg:
    #W1, W2,b1,b2 = slc.dynamicmodify(nm,x,sx,sy)
    gW1, gW2,gb1,gb2 = slcg.modifyThroughInterSection(ng,x,sx,sy, th1)
    gc = gc + 1
    #print(np.count_nonzero(slcm.D2))
    if np.count_nonzero(slcg.D2) < 45:
        print("Breaking at gc ", gc,np.count_nonzero(slcg.D2))
        slcg.first = True

hc = 0
for x in yh:
    #W1, W2,b1,b2 = slc.dynamicmodify(nm,x,sx,sy)
    hW1, hW2,hb1,hb2 = slch.modifyThroughInterSection(nh,x,sx,sy, th2)
    hc = hc + 1
    if np.count_nonzero(slch.D2) < 45:
        print("Breaking at hc ", hc,np.count_nonzero(slch.D2))
        slch.first = True
ic = 0
for x in yi:
    #W1, W2,b1,b2 = slc.dynamicmodify(nm,x,sx,sy)
    iW1, iW2,ib1,ib2 = slci.modifyThroughInterSection(ni,x,sx,sy, th1)
    ic = ic + 1
    #print(np.count_nonzero(slcm.D2))
    if np.count_nonzero(slci.D2) < 45:
        print("Breaking at ic ", ic,np.count_nonzero(slci.D2))
        slci.first = True

jc = 0
for x in yj:
    #W1, W2,b1,b2 = slc.dynamicmodify(nm,x,sx,sy)
    jW1, jW2,jb1,jb2 = slcj.modifyThroughInterSection(nj,x,sx,sy, th2)
    jc = jc + 1
    if np.count_nonzero(slcj.D2) < 45:
        print("Breaking at jc ", jc,np.count_nonzero(slcj.D2))
        slcj.first = True


# In[ ]:


#0
na.layers[1].set_weights([slca.D1,slca.d1])
na.layers[2].set_weights([slca.D2,slca.d2])
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


