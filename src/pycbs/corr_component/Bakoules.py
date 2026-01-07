def Bakoules_HF_extrapolation(Ec_X, Ec_Y, X, Y, beta=3.877):
    num = Ec_Y * ((X+1)**-beta) - Ec_X * ((Y+1)**-beta)
    den = ((X+1)**-beta) - ((Y+1)**-beta)
    return num/den

