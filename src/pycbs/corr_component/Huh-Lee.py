def Huh_Lee_HF_extrapolation(Ec_X, Ec_Y, X, Y, beta=0.220):
    num = Ec_Y * ((X+beta)**-3) - Ec_X * ((Y+beta)**-3)
    den = ((X+beta)**-3) - ((Y+beta)**-3)
    return num/den

