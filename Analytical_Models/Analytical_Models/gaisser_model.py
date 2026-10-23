import numpy as np

def gaisser_formula(E, theta):

    """
    dF/dSAdE = [m^-2*s^-1*sr^-1*GeV^-1]
    """

    return 10000 * 0.14 * E**-2.7 * ( (1/(1+((1.1*E*np.cos(theta))/115))) + (0.054/(1+((1.1*E*np.cos(theta))/850))) )

if __name__ == "__main__":
    pass
