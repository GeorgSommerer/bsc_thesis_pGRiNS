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
    return fld + (1 - fld) * (1 / (1 + (nod / thr) ** hill))
