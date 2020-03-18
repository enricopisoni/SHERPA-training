'''
Created on 18-nov-2016
downscale resolution, it works with different dimensions of the input file
@author: roncolato
'''
import numpy as np

def from7to28(Prec):
    step = 4;
    percNumDim = len(Prec.shape);
    
    if percNumDim == 2:
        d1 = Prec.shape[0];
        d2 = Prec.shape[1];
        p1 = int(d1/4);
        p2 = int(d2/4);
        Prec28 = np.zeros((p1, p2));
        
        for i in range(0, p1):
            for j in range(0, p2):
                inI = i*step;
                fiI = (i+1)*step;                
                inJ = j*step;
                fiJ = (j+1)*step;                
                sub = Prec[inI:fiI][:,inJ:fiJ];
                totNum = sub.shape[0] * sub.shape[1];
                Prec28[i,j] = sub.sum()/totNum;
                      
        return Prec28;

    if percNumDim == 3:
        d1 = Prec.shape[0];
        d2 = Prec.shape[1];
        d3 = Prec.shape[2];
        p1 = int(d1/4);
        p2 = int(d2/4);
        Prec28 = np.zeros((p1, p2, d3));
        
        for dimVar in range(0, d3):
            for i in range(0, p1):
                for j in range(0, p2):
                    inI = i*step;
                    fiI = (i+1)*step;                    
                    inJ = j*step;
                    fiJ = (j+1)*step;                    
                    sub = Prec[inI:fiI,inJ:fiJ,dimVar];
                    totNum = sub.shape[0] * sub.shape[1];
                    Prec28[i,j,dimVar] = sub.sum()/totNum;
                          
        return Prec28;

    if percNumDim == 4:
        d1 = Prec.shape[0];
        d2 = Prec.shape[1];
        d3 = Prec.shape[2];
        d4 = Prec.shape[3];
        p1 = int(d1/4);
        p2 = int(d2/4);
        Prec28 = np.zeros((p1, p2, d3, d4));
        
        for poll in range(0, d4):
            for dimVar in range(0, d3):
                for i in range(0, p1):
                    for j in range(0, p2):
                        inI = i*step;
                        fiI = (i+1)*step;                        
                        inJ = j*step;
                        fiJ = (j+1)*step;                        
                        sub = Prec[inI:fiI,inJ:fiJ,dimVar,poll];
                        totNum = sub.shape[0] * sub.shape[1];
                        Prec28[i,j,dimVar,poll] = sub.sum()/totNum;
                              
        return Prec28;
