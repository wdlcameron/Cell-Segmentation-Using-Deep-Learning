def calculate_anisotropy(image, Parameters):
    """
    Parameters contains:
        NA - Numerical Aperture
        index_of_refraction - Index of Refraction for the immersion media
        mag - Magnification
        gFactor - G-Factor for the detector
        
        kA - Correctional Factors for high NA
        kB - Correctional Factors for high NA
        kC - Correctional Factors for high NA
        
        
    image.shape[-1] should be equal to 3 with channels: Open, Parallel, Perpendicular
    """
    
    #image.shape[-1] should be equal to 3 with channels: Open, Parallel, Perpendicular 
    assert image.shape[2] == 3, f"Anisotropy image was expecting 3 channels, but got: {image.shape[2]}"
    
    
    Ka = Parameters.kA
    Kb = Parameters.kB
    Kc = Parameters.kC
    
    
    G = Parameters.gFactor
    
    
    anisotropy_image = np.zeros((image.shape))

    Para = image[:,:,1]
    Perp = image[:,:,2]


    if Parameters.numerical_aperture >=0.9:
        Ix = ((Kb * Para - Kc * Perp*G)/(Ka*Kb + Kb**2 -Ka*Kc - Kc**2))
        Iy = ((((Ka + Kb) * Perp*G) - (Ka+Kc)*Para)/(Ka*Kb+Kb**2-Ka*Kc-Kc**2))

    else:
        Iy = Para
        Ix = Perp
    
    anisotropy_image[:,:,0] = Iy
    anisotropy_image[:,:,1] = Ix
    anisotropy_image[:,:,2] = 0
    
    numerator = np.add(Iy, np.multiply(Ix, -G))
    denominator = np.add(Iy, np.multiply(Ix, 2*G))
       
        
    anisotropy_image[:,:,-1] = np.divide (numerator, denominator, where=denominator!=0)
    #anisotropy_image[:,:,2] = np.divide((np.add(Iy, np.multiply(Ix, -2*G))),(np.add(Iy, np.multiply(Ix, 2*G)))) 
    
    return (anisotropy_image)