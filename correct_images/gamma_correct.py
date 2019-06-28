import numpy as np

def gamma_correct(colour, bitdepth):
# translated to python by Jenny Walker jw22g14@soton.ac.uk

# MATLAB original code:
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # % Author: B.Thornton@soton.ac.uk
    # %
    # % gamma correction to account for non-linear peception of intensity by
    # % humans
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    colour = np.divide(colour, ((2**bitdepth-1)))

    if all(i < 0.0031308 for i in colour.flatten()):
        colour = 12.92*colour
    else:
        colour = 1.055 * np.power(colour, (1/1.5)) - 0.055

    colour = np.multiply(np.array(colour), np.array(2**bitdepth-1))
    return colour
