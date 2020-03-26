import  io
import numpy as np
from scipy import spatial
rs=[]
dataset = ['cậu tớ']
for i in dataset:
    s=i.split()
    u1 = s[0].strip() # word 1
    u2 = s[1].strip() # word 2
    print(u1)
    print(u2)
    sim = (2 - spatial.distance.cosine(u1, u2))/2
    rs.append(sim)

 
