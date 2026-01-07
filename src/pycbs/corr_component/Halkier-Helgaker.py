def Halkier_corr_extrapolation(Ec_X, Ec_Y, X, Y):
    num = Ec_X * (X**3) - Ec_Y * (Y**3)
    den = (X**3) - (Y**3)
    return num/den