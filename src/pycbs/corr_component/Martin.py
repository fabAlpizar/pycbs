def Martin_HF_extrapolation(Ec_X, Ec_Y, X, Y, beta=3.315):
    num = Ec_Y * ((X+0.5)**-beta) - Ec_X * ((Y+0.5)**-beta)
    den = ((X+0.5)**-beta) - ((Y+0.5)**-beta)
    return num/den

