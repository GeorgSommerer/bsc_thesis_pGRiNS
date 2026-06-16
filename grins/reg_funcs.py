# Positive Shifted Hill function
def psH(nod, fld, thr, hill):
    """
    Positive Shifted Hill function.

    Parameters
    ----------
    nod : float
        The node expression value.
    fld : float
        The fold change.
    thr : float
        The half-maximal threshold value.
    hill : float
        The hill coefficient.

    Returns
    -------
    float
        The value of the Positive Shifted Hill function.
    """
    """
    # Idea: pert_factor reduces originally sampled parameter range to 1/pert_factor% (e.g. if pert_factor = 100, then range is 1% of original range)
    # For CRISPR-KO, pert_scalar = 0, and the whole equations should be zero, so that Prod(H) is 0
    # For CRISPRa (pert_scalar > 1), pert_scalar = pert_factor
        # ActFld of psH/activating edges should go to top 1/pert_factor% (e.g from [1,100] to [99.01,100]) -> divide by pert_scalar
            # Set a new minimum val and perform min-max transformation to new interval
        # InhFld of nsH/inhibiting edges should go to bottom 1/pert_factor% (e.g. from [0.01,1] to [0.01,0.0199])
            # Set a new maximum val and perform min-max transformation to new interval
    # For CRISPRi (pert_scalar < 1), pert_scalar = 1/pert_factor, and the opposite should happen -> multiply by pert_scalar
    # If pert_scalar = 1 (usual case, no perturbation) nothing changes
    
    if pert_scalar == 0:
        return 0
    else:
        max_val = 100
        min_val = 1
        if pert_scalar > 1:
            min_val = max_val-(max_val-min_val)/pert_scalar
            fld = (fld-min_val)/(max_val-min_val)
        elif pert_scalar < 1:
            max_val = (max_val-min_val)*pert_scalar
            fld = (fld-min_val)/(max_val-min_val)
        return (fld + (1 - fld) * (1 / (1 + (nod / thr) ** hill))) #/ fld <- why divide there???
    """
    """
    if pert_scalar == 0:
        return 0
    else:
        fld *= pert_scalar
    """
    return (fld + (1 - fld) * (1 / (1 + (nod / thr) ** hill))) / fld


# Negative Shifted Hill function
def nsH(nod, fld, thr, hill):
    """
    Negative Shifted Hill function.

    Parameters
    ----------
    nod : float
        The node expression value.
    fld : float
        The fold change.
    thr : float
        The half-maximal threshold value.
    hill : float
        The hill coefficient.

    Returns
    -------
    float
        The value of the Negative Shifted Hill function.
    """
    """
    if pert_scalar == 0:
        return 0
    else:
        max_val = 1
        min_val = 0.01
        if pert_scalar < 1:
            min_val = max_val-(max_val-min_val)*pert_scalar
            fld = (fld-min_val)/(max_val-min_val)
        elif pert_scalar > 1:
            max_val = (max_val-min_val)/pert_scalar
            fld = (fld-min_val)/(max_val-min_val)
        return fld + (1 - fld) * (1 / (1 + (nod / thr) ** hill))
    """
    """
    if pert_scalar == 0:
        return 0
    else:
        fld /= pert_scalar
    """
    return fld + (1 - fld) * (1 / (1 + (nod / thr) ** hill))
