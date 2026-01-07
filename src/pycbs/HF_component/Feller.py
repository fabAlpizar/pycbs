import math as mt
def Feller_HF_extrapolation(Ehf_X, Ehf_Y, X, Y, alfa=1.353):
    num = Ehf_Y * mt.exp(-alfa * X ) - Ehf_X * mt.exp(-alfa * Y)
    den = mt.exp(-alfa * X ) - mt.exp(-alfa * Y)
    return num/den

