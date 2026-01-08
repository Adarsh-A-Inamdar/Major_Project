# src/pso_tune.py
import numpy as np, random

def quick_train_and_eval(lr, wd, bs, head):
    # Your training and evaluation logic here
    # Return a float between 0 and 1 representing validation accuracy
    val_acc = 0.5  # placeholder
    return val_acc


def clip(v, lo, hi): return max(lo, min(hi, v))

def decode(pos):
    lr   = 10**np.interp(pos[0],[0,1],[np.log10(1e-5), np.log10(1e-2)])
    wd   = np.interp(pos[1],[0,1],[0.0,1e-2])
    bs   = [16,32,64][int(pos[2]*3)%3]
    head = [128,256,512][int(pos[3]*3)%3]
    return lr,wd,bs,head

N=10; D=4
pos=np.random.rand(N,D); vel=np.zeros((N,D))
pbest=pos.copy(); pbest_val=np.zeros(N)-1
gbest=None; gbest_val=-1

for it in range(15):
    for i in range(N):
        lr,wd,bs,head=decode(pos[i])
        val_acc=quick_train_and_eval(lr,wd,bs,head)  # implement short run
        if val_acc>pbest_val[i]: pbest_val[i]=val_acc; pbest[i]=pos[i].copy()
        if val_acc>gbest_val: gbest_val=val_acc; gbest=pos[i].copy()
    w,c1,c2=0.6,1.4,1.4
    for i in range(N):
        r1,r2=np.random.rand(D),np.random.rand(D)
        vel[i]=w*vel[i]+c1*r1*(pbest[i]-pos[i])+c2*r2*(gbest-pos[i])
        pos[i]=np.array([clip(p+v,0,1) for p,v in zip(pos[i],vel[i])])
print("Best PSO val acc:", gbest_val)
