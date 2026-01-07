import math as mt
def Klopper_HF_extrapolation(Ehf_X, Ehf_Y, X, Y, alfa=4.257):
    num = Ehf_X * mt.exp(-alfa * mt.sqrt(Y) ) - mt.exp(-alfa * mt.sqrt(X)) * Ehf_Y
    den = mt.exp(-alfa * mt.sqrt(X)) - mt.exp(-alfa * mt.sqrt(Y))
    return num/den

