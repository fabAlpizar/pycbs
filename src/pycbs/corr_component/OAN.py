def OAN_corr_extrapolation(Ec_X, Ec_Y, beta=2.086):
    num = Ec_Y * 27 - (beta**3) * Ec_X
    den = 27 - (beta**3)
    return num/den