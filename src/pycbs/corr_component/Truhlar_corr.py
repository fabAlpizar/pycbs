def Truhlar_corr_extrapolation(Ec_X, Ec_Y, X, Y, beta=2.751):
    num = Ec_Y * (X**-beta) - Ec_X * (Y**-beta)
    den = (X**-beta) - (Y**-beta)
    return num/den