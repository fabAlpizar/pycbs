def Truhlar_HF_extrapolation(Ehf_X, Ehf_Y, X, Y, alfa=3.337):
    num = (Ehf_Y * X**-alfa) - (Ehf_X * Y**-alfa)
    den = (X**-alfa) - (Y**-alfa)
    return num/den

