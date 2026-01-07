import math as mt
def Jensen_HF_extrapolation(Ehf_X, Ehf_Y, X, Y, alfa=5.163):
    num = Ehf_Y * (X+1) * mt.exp(-alfa * mt.sqrt(X) ) - Ehf_X * (Y+1) * mt.exp(-alfa * mt.sqrt(Y) )
    den = (X+1) * mt.exp(-alfa * mt.sqrt(X) ) - (Y+1) * mt.exp(-alfa * mt.sqrt(Y) )
    return num/den

