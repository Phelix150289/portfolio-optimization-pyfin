import pandas as pd

def get_factor_ret():
    
    """
    Import FF 3 factor returns and merge with Momemtum factor
    
    """
    # Process FF 3 factor returns
    ff3 = pd.read_csv('data/F-F_Research_Data_Factors.csv', index_col = 0) / 100
    ff3.index = pd.to_datetime(ff3.index, format = '%Y%m')
    ff3.index = ff3.index.to_period('M')
    
    # Process MOM returns
    mom = pd.read_csv('data/F-F_Momentum_Factor.csv', index_col = 0) /100
    mom.index = pd.to_datetime(mom.index, format = '%Y%m')
    mom.index = mom.index.to_period('M')
    # Join factor data
    factors = ff3.join(mom, how="inner")
    return factors


def get_ind_ret():
    """
    Imports FF 48 industry portfolio returns from a saved file.
    
    Inputs
    ===========
    No input parameters required
    
    Returns
    ===========
    ret: a DataFrame that stores the returns data of each portfolio in FF 48 industry portfolios
    """
    
    ret = pd.read_csv('data/FF_48_portfolios_ret_vw.csv', header=0, index_col=0, na_values = -99.99)/100
    ret.index = pd.to_datetime(ret.index, format="%Y%m")
    ret.index = ret.index.to_period('M')
    ret.columns = ret.columns.str.strip() # to remove the annoying white space at the end of some industry names.
    
    return ret

def get_ind_capweight():
    """
    Import the size and number of firms in FF 48 industry portfolios from saved files and compute the market cap weight for each portfolio.
    
    Inputs
    ===========
    No input parameters required
    
    Returns
    ===========
    capweight: a DataFrame that stores the market cap weight of each portfolio in FF 48 industry portfolios
    """
    
    size = pd.read_csv('data/FF_48_portfolios_avgsize.csv', header=0, index_col=0, na_values = -99.99)
    size.index = pd.to_datetime(size.index, format="%Y%m")
    size.index = size.index.to_period('M')
    size.columns = size.columns.str.strip() 
 
    firmnum = pd.read_csv('data/FF_48_portfolios_firmnum.csv', header=0, index_col=0)
    firmnum.index = pd.to_datetime(firmnum.index, format="%Y%m")
    firmnum.index = firmnum.index.to_period('M')
    firmnum.columns = firmnum.columns.str.strip() 
    
    mktcap = size*firmnum
    capweight = mktcap.div(mktcap.sum(axis=1), axis=0)
    return capweight

