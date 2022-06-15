from math import log, sqrt, pi, exp, erf
from scipy.stats import norm
from datetime import datetime, date
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import itertools
import datetime, time
from copy import deepcopy

rfrs = {'sETHnew': 0.06, 'sSOLnew': 0.06, 'sLINKnew':0.06};

final_spots_ETH= {'1637308800': 4099.76312982,
 '1639123200': 4044.35,
 '1637913600': 4302.85,
 '1638518400': 4569.15,
 '1641542400': 3202.75,
 '1639728000': 3885.55, #DONE HERE
 '1640332800': 4093.26,
 '1640937600': 3724.59,#DONE HERE
 '1643961600': 2818.24,
 '1642752000': 2892.97,
 '1643356800': 2404.98,
 '1642147200': 3253.18,
 '1646380800': 2732.30,
 '1644566400': 3100.49317255,
 '1645171200': 2905.57,
 '1645776000': 2613.09,
 '1648800000': 3278.82,
 '1647590400': 2795.09,#Round 7 onwards begins here
 '1646985600': 2609.55,
 '1648195200': 3132.81,
 '1649404800': 3282.58, 
 '1650009600': 3041.57,
 '1650614400': 3021.1541,
 '1651219200': 2923.06529924,
 '1651824000': 2726.19538135, 
 '1652428800': 2049.14930423, 
 '1653033600': 2024.42660286, 
 '1653638400':1762.77188149}

final_spots_SOL = {'1646380800': 91.93384197, 
                   '1646985600': 82.29, 
                   '1647590400': 85.746802, 
                   '1648195200': 102.99,
                   '1648800000': 126.4,
                   '1649404800': 119.04054787, 
                   '1650009600': 101.71586655,
                   '1650614400': 102.46891205,
                   '1651219200': 97.77,
                   '1651824000': 82.34196421, 
                   '1652428800': 49.25745707, 
                   '1653033600': 52.14682904, 
                   '1653638400': 41.0}

final_spots_BTC = {'1637308800':56531.90592295,
 '1639123200': 47748.0,
 '1637913600': 56771.648,
 '1638518400': 56705.83721265,
 '1641542400': 41713.85470696,
 '1639728000': 47028.20351278, #DONE HERE
 '1640332800': 51025.63,
 '1640937600': 47174.75092386,#DONE HERE
 '1643961600': 37905.42378934,
 '1642752000': 39212.98063389,
 '1643356800': 36938.8739981,
 '1642147200': 42511.48048572,
 '1646380800': 41418.496,
 '1644566400': 43426.63586272,
 '1645171200': 40681.364,
 '1645776000': 38478.225,
 '1648800000': 45035.42,
 '1647590400': 40578.62638551,
 '1646985600': 39148.36528964,
 '1648195200': 43957.95}

final_spots_LINK = {'1637308800':27.32163099,
 '1639123200': 19.48025253,
 '1637913600': 25.29112656,
 '1638518400': 25.13417066,
 '1641542400': 23.52762984,
 '1639728000': 18.99273426, #DONE HERE
 '1640332800': 22.21905094,
 '1640937600': 19.93987203,#DONE HERE
 '1643961600': 16.34697007,
 '1642752000': 19.65216638,
 '1643356800': 15.09543454,
 '1642147200': 24.67979446,
 '1646380800': 14.34,
 '1644566400': 17.50620141,
 '1645171200': 15.87605758,
 '1645776000': 13.35222471,
 '1648800000': 16.82333912,
 '1647590400': 14.43,
 '1646985600': 13.31,
 '1648195200': 16.043429, # Round 7 begins here
 '1649404800': 15.75, 
 '1650009600': 13.91864758,
 '1650614400': 13.81,
 '1651219200': 12.62,
 '1651824000': 10.81172, 
 '1652428800': 7.11, 
 '1653033600': 7.04103372, 
 '1653638400': 6.4264725}

final_spots = {"sETHnew": final_spots_ETH, 'sBTCnew': final_spots_BTC, 'sLINKnew':final_spots_LINK, 'sSOLnew': final_spots_SOL}

starting_sUSD_ETH= [10.66*1000000, 24.83*1000000, 30.0*1000000, 19.7*1000000, 27.4*1000000, 21.2*1000000, 20.6 * 1000000, 20.56 * 1000000]
starting_sUSD_BTC = [10.58*1000000, 24.73*1000000,28.36*1000000, 18.54*1000000, 10.19*1000000, 8.28*1000000, 7.04*1000000]
starting_sUSD_LINK = [2.45*1000000, 5.22*1000000, 4.25*1000000, 4.17*1000000, 3.33*1000000, 2.76*1000000, 1.5*1000000, 1.74*1000000] 

starting_sUSD_SOL = [304100, 4.98*1000000, 1.48 *1000000, 2.92 * 1000000]

starting_sUSDs = {"sETHnew": starting_sUSD_ETH, "sBTCnew": starting_sUSD_BTC, "sLINKnew": starting_sUSD_LINK, 'sSOLnew': starting_sUSD_SOL}

pnls_real_ETH  = [-2.074,.001,-.626,-.7,.36,2.55]
pnls_real_BTC = [-0.586, -0.1164, -0.2137, -0.3414, 0.667, 1.185]
pnls_real_LINK = [-1.906, -2.48, 2.846, -4.5, 0.834, 0.0713]
pnls_real_SOL = [+1.038, +0.1855, 0.631, -5.4]
pnls_reals = {"sETHnew": pnls_real_ETH, "sBTCnew": pnls_real_BTC, "sLINKnew": pnls_real_LINK, 'sSOLnew': pnls_real_SOL}



from Delta_Cutoffs import *

def UpdateOtherListings(lists, df, K, EXP,  drops):
    expiries = df.expiry.unique()
    for i in range(0,len(lists)):
        for exp in expiries:
            strikes = df[df.expiry == exp].strike.unique()
            for strike in strikes:
                dic = lists[i]
                if ((strike == K) and (exp == EXP)) and (i in drops):
                    pass
                else:
                    dic[str(exp)][str(strike)].append(dic[str(exp)][str(strike)][-1])
    return lists
    
def Kill_Expired_Pos(lists, df, t):
    expiries = df.expiry.unique()
    for exp in expiries:
        strikes = df[df.expiry == exp].strike.unique()
        
        for strike in strikes:
            if t > exp:
                for dic in lists:
                    dic[str(exp)][str(strike)][-1] = 0
    return lists
                       
def sign_maker(x):
    if x['OpenClose'] == "Open":
        OC_sign = 1
    elif x['OpenClose'] == "Close":
        OC_sign = -1
    
    if (x['tradeType'] == "LONG_CALL") | (x['tradeType'] == "LONG_PUT"):
        prem_sign = 1
        position_sign = -1
    
    elif (x['tradeType'] == "SHORT_CALL") | (x['tradeType'] == "SHORT_PUT"):
        prem_sign = -1
        position_sign = 1
        
    return OC_sign * prem_sign, OC_sign * position_sign
    
def Premium_Finder(x,asset):
    t_norm = (60*60*24*365);
    
    prem_sign, pos_sign = sign_maker(x)
    n = x['amount']
    if (x['tradeType'] == 'LONG_CALL') | (x['tradeType'] == 'SHORT_CALL'):
        return prem_sign * n *bs_call(x['baseIv']*x['skew'], x['strike'],x['price'],rfrs[asset], x['tau']/t_norm)
    elif (x['tradeType'] == 'LONG_PUT')| (x['tradeType'] == 'SHORT_PUT'):
        return prem_sign * n *bs_put(x['baseIv']*x['skew'], x['strike'],x['price'],rfrs[asset], x['tau']/t_norm)
    
    
def Spot_Fee(x, SNX_fee = 0.001):
    return abs(x['amount'])* x['price']*SNX_fee

def Option_Fee(x, Option_fee = 0.01):
    return x['Premiums']*Option_fee

def FeeCombiner(x):
    if x['Premiums'] > 0:
        return x['Premiums'] + x['SpotFees'] + x['OptionFees']
    elif x['Premiums'] < 0: 
        return x['Premiums'] + x['SpotFees'] - x['OptionFees']

def VU_fee(x):
    
    if x['Premiums']>0:
        return max(0,x['TrueCost'] - x['TotalCost'])
    elif x['Premiums'] < 0:
        return max(0,-x['TotalCost'] - x['TrueCost'])
    
def Round_Analyser_Fees(df, asset):
    new_df = df.copy()
    
    fee_prem_df = pd.DataFrame();
    
    new_df = new_df.drop(['trader', 'txHash','block','callDelta','putDelta'], axis = 1)
    #new_df["amount"], new_df["tradeType2"] = zip(*new_df.apply(CALL_PUT, axis=1))
    #new_df['amount'] = new_df.apply(OpenCloseFlip, axis =1)
    
    fee_prem_df['Premiums'] = new_df.apply(lambda x: Premium_Finder(x,asset), axis = 1)
    new_df['Premiums'] = new_df.apply(lambda x: Premium_Finder(x,asset), axis = 1)
    fee_prem_df['amount'] = new_df['amount']
    fee_prem_df['SpotFees'] = new_df.apply(lambda x: Spot_Fee(x), axis = 1)
    fee_prem_df['OptionFees'] = fee_prem_df.apply(lambda x: Option_Fee(x), axis = 1)
    fee_prem_df['tradeType2'] = new_df['tradeType']
    fee_prem_df['TotalCost'] = fee_prem_df.apply(FeeCombiner, axis = 1)
    fee_prem_df['TrueCost'] = new_df.apply(true_cost_sign, axis = 1)
    fee_prem_df['VUFee'] = fee_prem_df.apply(VU_fee, axis = 1)

    return fee_prem_df 

def TimeSeriesInstantMaker(df):
    new_df = df.copy()
    
    expiries = new_df.expiry.unique()
    
    call_collateral_list = {};
    put_collateral_list = {};
    
    pos_call_list = {};
    pos_put_list = {};
    
    all_data =[call_collateral_list, put_collateral_list, pos_call_list, pos_put_list] 
    for exp in expiries: 
        
        call_collateral_list[str(exp)] = {}
        put_collateral_list[str(exp)] = {}
        
        pos_call_list[str(exp)] = {}
        pos_put_list[str(exp)] = {}
        
        strikes = new_df[new_df.expiry == exp].strike.unique()
        
        for strike in strikes:
            call_collateral_list[str(exp)][str(strike)] = [0]
            put_collateral_list[str(exp)][str(strike)] = [0]
            
            pos_call_list[str(exp)][str(strike)] = [0]
            pos_put_list[str(exp)][str(strike)] = [0]
    
    
            
    for i in range(0,len(df)):
        ele = df.iloc[i];
        exp = ele['expiry']
        strike = ele['strike']
        t = ele['timestamp']
        
        if ele['OpenClose'] == "Open":
            sign = 1
        elif ele['OpenClose'] == 'Close':
            sign = -1
         
        # Update the collateral required when users open Long Calls/Puts
        # AND update the position for each trade
        
        if ele['tradeType'] == "LONG_CALL":
            x = all_data[0][str(exp)][str(strike)]#call_collateral_list[str(exp)][str(strike)]
            y = all_data[2][str(exp)][str(strike)]#pos_call_list[str(exp)][str(strike)]
            x.append(x[-1] + sign * ele['amount']) 
            y.append(y[-1] + sign * ele['amount']) 
            drops = [0,2]
            all_data = UpdateOtherListings(all_data, new_df,strike, exp, drops)
        
        elif ele['tradeType'] == "LONG_PUT":
            x = all_data[1][str(exp)][str(strike)]#put_collateral_list[str(exp)][str(strike)]
            y = all_data[3][str(exp)][str(strike)]#pos_put_list[str(exp)][str(strike)]
            x.append(x[-1] + sign * ele['amount']) 
            y.append(y[-1] + sign * ele['amount'])
            drops = [1,3]
            all_data = UpdateOtherListings(all_data, new_df,strike, exp,drops)
            
        elif ele['tradeType'] == "SHORT_CALL":
            x = all_data[2][str(exp)][str(strike)]#pos_call_list[str(exp)][str(strike)]
            drops = [2]
            x.append(x[-1] + (- sign * ele['amount'])) 
            all_data = UpdateOtherListings(all_data, new_df,strike, exp, drops)
            
        elif ele['tradeType'] == "SHORT_PUT":
            x = all_data[3][str(exp)][str(strike)]#pos_put_list[str(exp)][str(strike)]
            drops = [3]
            x.append(x[-1] + (- sign * ele['amount']))
            all_data = UpdateOtherListings(all_data, new_df,strike, exp, drops)
        #print(all_data[1])
        all_data = Kill_Expired_Pos(all_data, new_df, t)
       
            
    ret_data = {"BaseCollateral": call_collateral_list, "QuoteCollateral": put_collateral_list, "CallPositions": pos_call_list, "PutPositions":pos_put_list} 
    return ret_data

def BaseCollateral_TimeSeries(df,df_data):
    df_collat = df['BaseCollateral']
    all_collats = []
    exps = list(df_collat.keys())
    for exp in exps:
        strikes = list(df_collat[exp].keys())
        for strike in strikes:
            all_collats.append(np.array(df_collat[exp][strike]))
        
    return sum(all_collats)
 
    
def QuoteCollateral_TimeSeries(df,df_data):
    df_collat = df['QuoteCollateral']
    all_collats = []
    exps = list(df_collat.keys())
    for exp in exps:
        strikes = list(df_collat[exp].keys())
        for strike in strikes:
            all_collats.append(int(strike)*np.array(df_collat[exp][strike]))
        
    return sum(all_collats)

def bad_index_finder(df):
    exps = np.sort(np.array(df.expiry.unique()))[:-1]
    bad_indexs = {}
    for exp in exps:
        
        diff = df['timestamp'] - exp
        bad_index = diff[diff <0 ].idxmax()
        bad_indexs[str(bad_index+1)] = str(exp)
    return bad_indexs


def Collat_TimeSeries(collat_df, data_df,asset):
    spot_series = data_df['price']
    
    base_collateral_series = BaseCollateral_TimeSeries(collat_df,data_df)
    
    quote_collateral_series = QuoteCollateral_TimeSeries(collat_df,data_df)
    
    AMM_collat_fees = []
    
    for i in range(1,len(base_collateral_series)):
        CHANGE_base_collat = base_collateral_series[i] - base_collateral_series[i-1]
        AMM_collat_fee = abs(CHANGE_base_collat * 0.001 * spot_series[i-1])
        AMM_collat_fees.append(AMM_collat_fee)
    
    base_collateral_value_series = spot_series* base_collateral_series[1:]
    collat_costs = [];
    
    
    BAD_indexes = bad_index_finder(data_df);
    
    for i in range(0,len(base_collateral_series[1:])):
        current_base = base_collateral_series[1:][i]
        if i == 0:
            dbase = current_base
        elif i > 0:
            dbase = current_base - base_collateral_series[1:][i-1]
        
        if str(i) in list(BAD_indexes.keys()):
            spot = final_spots[asset][str(BAD_indexes[str(i)])]
            
        spot = spot_series[i]  
        paid_to_collat = dbase*spot
        collat_costs.append(paid_to_collat)
                
    
    data_ret = {"Timestamp": data_df['timestamp'],
                "BaseCollateral": base_collateral_series[1:],
                "BaseCollateralValue": base_collateral_value_series,
                "QuoteCollateral": quote_collateral_series[1:],
                "CostToCollat":collat_costs,
                "SNXCollateralFees": AMM_collat_fees,
               "Spot":spot_series}

    return pd.DataFrame(data_ret)
def UpdateVols(base_dic,skew_dic, df, K, EXP):
    expiries = df.expiry.unique()
    for exp in expiries:
        strikes = df[df.expiry == exp].strike.unique()
        
        if exp == EXP:
            pass
        else: 
            base_dic[str(exp)].append(base_dic[str(exp)][-1])
            
        for strike in strikes:
            if ((strike == K) and (exp == EXP)):
                pass
            else:
                skew_dic[str(exp)][str(strike)].append(skew_dic[str(exp)][str(strike)][-1])
    return base_dic, skew_dic


def VolSurfaceSeries(df):
    expiries = df.expiry.unique()
    
    skews = {};
    baseIvs = {};
    
    for exp in expiries: 

        first_baseIv = (df[df.expiry == exp]['baseIv']).iloc[0]

        baseIvs[str(exp)] = [first_baseIv]
        skews[str(exp)] = {}
        
        strikes = df[df.expiry == exp].strike.unique()
        
        for STRIKE in strikes:

            firstskew = df[(df.expiry == exp) & (df.strike == STRIKE)]['skew'].iloc[0]
            skews[str(exp)][str(STRIKE)] = [firstskew]
        
    for i in range(0,len(df)):
        ele = df.iloc[i]
        strike_use = ele['strike']
        exp_use = ele['expiry']
            
        new_baseIv = ele['baseIv']
        new_skew = ele['skew']
             
        baseIvs[str(exp_use)].append(new_baseIv)
        skews[str(exp_use)][str(strike_use)].append(new_skew)
            
        baseIvs, skews = UpdateVols(baseIvs,skews, df, strike_use, exp_use)    
    
    return baseIvs, skews

def OptionExposure(baseIvs, Skews, Positions, df, index, rfr = 0.06):
    timestamp = df['timestamp'].iloc[index]
    spot = df['price'].iloc[index]
    exps = np.sort(np.array([int(ele) for ele in list(Positions['CallPositions'].keys())]))
    
    option_val = 0
    delta_val = 0
    
    for exp in exps:
        strikes = np.sort(np.array([int(ele) for ele in list(Positions['CallPositions'][str(exp)].keys())]))
        for strike in strikes:
            skew_use = Skews[str(exp)][str(strike)][1:][index]
            base_use = baseIvs[str(exp)][1:][index]
            vol_use = skew_use * base_use
            
            call_pos = Positions['CallPositions'][str(exp)][str(strike)][1:][index]
            put_pos = Positions['PutPositions'][str(exp)][str(strike)][1:][index]

            
            if timestamp < exp:
                call_val = call_pos * bs_call(vol_use, strike, spot, rfr, (exp - timestamp)/(3600*24*365))
                put_val = put_pos * bs_put(vol_use, strike, spot, rfr, (exp - timestamp)/(3600*24*365))
                
                delta_call_val = call_pos * call_delta(vol_use, strike, spot, rfr, (exp - timestamp)/(3600*24*365)) 
                delta_put_val = put_pos * put_delta(vol_use, strike, spot, rfr, (exp - timestamp)/(3600*24*365))
            else: 
                call_val = 0
                put_val = 0
                delta_call_val = 0 
                delta_put_val = 0
            
            net_val = call_val + put_val
            delta_val += delta_call_val + delta_put_val 
            
            option_val += net_val
    
    return option_val,delta_val

def true_cost_sign(x):
    if (x['tradeType'] == "LONG_CALL") | (x['tradeType'] == "LONG_PUT"):
        sign = +1
    elif (x['tradeType'] == "SHORT_CALL") | (x['tradeType'] == "SHORT_PUT"):
        sign = -1
    
    if x['OpenClose'] == "Open":
        sign2 = 1;
    elif x['OpenClose'] == "Close":
        sign2 = -1
        
    return x['totalCost']*sign * sign2


def expiry_flipper(x):
    if x['OpenClose'] == 'Open':
        sign = 1
    else: 
        sign = -1
    if x['tradeType'] == "LONG_CALL":
        return -sign * x['amount']
    elif x['tradeType'] == "LONG_PUT":
        return -sign *x['amount']
    elif x['tradeType'] == 'SHORT_CALL':
        return sign *x['amount']
    elif x['tradeType'] == 'SHORT_PUT':
        return sign *x['amount']
    
def change_trade_type(x):
    if (x['tradeType'] == "LONG_CALL") | (x['tradeType'] == "SHORT_CALL"):
        return "CALL"
    elif (x['tradeType'] == "LONG_PUT") | (x['tradeType'] == "SHORT_PUT"):
        return "PUT"
    
def expPNL(x, S):
    if x['tradeType'] == 'CALL':
        return x['signedamt'] * max(S - x['strike'], 0 )
    
    elif x['tradeType'] == 'PUT':
        return x['signedamt'] *max(x['strike'] - S, 0)
    
def Option_Expiry_PNL(df, exp, asset):
    df_edit = df.copy()
    exp_spot = final_spots[asset][str(exp)]
    df_edit['signedamt'] = df_edit.apply(lambda x: expiry_flipper(x),axis =1)
    df_edit['tradeType'] = df_edit.apply(lambda x: change_trade_type(x),axis =1)
    df_edit = df_edit[df_edit.expiry == exp].groupby(['strike','tradeType'])['signedamt'].sum().reset_index()
    df_edit['expPNL'] = df_edit.apply(lambda x: expPNL(x,exp_spot),axis = 1) 
    return df_edit
def FinalCollatPNL(df,exps, asset):
    cost_to_collat = df['CostToCollat'].sum()
    final_collat = df['BaseCollateral'].iloc[-1]
    final_spot = final_spots[asset][str(exps[-1])]
    final_collat_val = final_collat * final_spot
    snx_fees = df['SNXCollateralFees'].sum()
    pnl = -cost_to_collat + final_collat_val
    return pnl, snx_fees 

def entire_round_pnl(df1, asset, chosen_round):
    df = df1.copy()
    use_All_Collat_Pos_Data = TimeSeriesInstantMaker(df[chosen_round])
    use_collat_df = Collat_TimeSeries(use_All_Collat_Pos_Data,df[chosen_round],asset)
    premiums_df = Round_Analyser_Fees(df[chosen_round], asset)
    exps = np.sort(df[chosen_round].expiry.unique())
    pnl_exps = []
    for exp in exps:
        option_pnl = Option_Expiry_PNL(df[chosen_round], exp, asset)['expPNL'].sum()
        pnl_exps.append(option_pnl)
        
    use_collat_pnl, use_snx_fees = FinalCollatPNL(use_collat_df, exps,asset)
    
    all_pnl = premiums_df['TrueCost'].sum() + sum(pnl_exps) + use_collat_pnl - use_snx_fees
    return all_pnl, [premiums_df['TrueCost'].sum(), sum(pnl_exps), use_collat_pnl, use_snx_fees],use_collat_df

def all_rounds_combiner(df, asset, num_rounds):
    net_pnls = []
    all_data = []
    for i in range(0,num_rounds):
        pnl = entire_round_pnl(df,asset,i)
        net_pnls.append(100*pnl[0]/starting_sUSDs[asset][i])
        all_data.append(pnl)
    return net_pnls, all_data



def Hedge(dfc, df_collat, L0,asset, SHIFT, hedge_freq, hedge_cost = 0.001):
    liq_max = L0/3;
    df = dfc.copy()
    error_dt = []
    HEDGE_TIMES = []
    #hedge_liqs = [0]; #How much free liquidity is left
    hedge_time = df.iloc[0]['timestamp'] + hedge_freq * 60 * 60 + SHIFT *  60 * 60#when is the AMM going to hedge?
    
    hedge_X = [[0,0]]
    short_cost = [0]
    hedge_fees = [0]
    pres_delta = [0]
    dX = [0]
    FREELIQs = []
    freeliq = liq_max
    for i in range(0,len(df)):
        
        S = df.iloc[i]['price']
        t = df.iloc[i]['timestamp']
        orig_len = len(hedge_X);
        if i==0:
            S_old = 0;
        else:
            S_old = df.iloc[i-1]['price']
        #freeliq += (S_old - S) * hedge_X[i][0]
        if t < hedge_time: #Don't hedge, just update everything
            

                
            #
            #hedge_liqs.append(hedge_liqs[i])
            hedge_X.append(hedge_X[i])
            hedge_fees.append(0)
            pres_delta.append(pres_delta[i])
            dX.append(0)
            error_dt.append(0)
            
            FREELIQs.append(freeliq)

        elif t >= hedge_time: 
            
            #print(len(hedge_X)) 
            HEDGE_TIMES.append((t-df['timestamp'].iloc[0])/3600)
            if i==0:
                S_old = 0;
            else:
                S_old = df.iloc[i-1]['price']
            time_error = (t-hedge_time)/3600
            error_dt.append(time_error)
            
            if time_error <= .01:
                S = df.iloc[i]['price']
            elif time_error > .01:
                #print([df.iloc[i]['price'],df.iloc[i-1]['price']])
                S = df.iloc[i]['price']
                #S = (df.iloc[i]['price'] + df.iloc[i-1]['price'])/2
            
            shortRatio = 2 
            hedge_time += hedge_freq * 60 * 60;
            present_net_delta = -df.iloc[i]['netDelta']  + hedge_X[i][0] + df_collat.iloc[i]['BaseCollateral']
            
            eth_needed = - present_net_delta
            pres_delta.append(eth_needed)
            
            
                
            
            
            current_hedged_X = hedge_X[i][0]
   #UPDATE HERE         
            
            
            if (current_hedged_X >= 0) and (eth_needed > 0): #need to buy more ETH to hedge
                
                newfreeliq = freeliq - abs(eth_needed) * S #find how much total liquidity is needed to hedge
                
                if newfreeliq > 0: #if AMM has enough liquidity
                    #hedge_liqs.append(hedge_liqs[i] + abs(eth_needed)*S) #AMM locks eth_needed * S to long more eth 
                    
                    hedge_X.append([hedge_X[i][0]+eth_needed, S]) #find the new delta of the pool (should be 0)
                    
                    
                    hedge_fees.append(abs(eth_needed)*S * hedge_cost) #find the fees for hedging
                    
                    dX.append(eth_needed)
                    
                    newfreeliq = freeliq - abs(eth_needed) * S
                    
                    FREELIQs.append(newfreeliq)
                    freeliq = newfreeliq
                    
                else:
                    #hedge_liqs.append(hedge_liqs[i]) #AMM does not have enough funds to hedge
                    hedge_X.append([hedge_X[i][0], S])
                    hedge_fees.append(0)
                    dX.append(0)
                    
                    FREELIQs.append(freeliq)
                    

            elif current_hedged_X >= 0 and eth_needed < 0 and current_hedged_X + eth_needed > 0: 
                #AMM is currently long eth to hedge, but needs to short (will remain net long)
                
                #hedge_liqs.append(hedge_liqs[i] + abs(eth_needed) * S) #AMM will sell some eth off at the current price
                hedge_X.append([hedge_X[i][0]+eth_needed, S]) #Update the delta 
                hedge_fees.append(abs(eth_needed) * S * hedge_cost) 
                dX.append(eth_needed)
                newfreeliq = freeliq + abs(eth_needed) * S
                FREELIQs.append(newfreeliq)
                freeliq = newfreeliq
                    
            elif current_hedged_X >= 0 and eth_needed < 0 and current_hedged_X + eth_needed < 0: 
                
                #AMM is currently long eth to hedge, needs to short so will be net short

                #hedge_liqs.append(hedge_liqs[i] - abs(current_hedged_X) * S + shortRatio * abs(eth_needed + current_hedged_X) * S) 
                # Unlock liquidity from the closed longs, lock 2x for shorts 
                hedge_X.append([hedge_X[i][0]+eth_needed, S])
                dX.append(eth_needed)
                hedge_fees.append((abs(eth_needed) + current_hedged_X)* S * hedge_cost) 
                newfreeliq = freeliq - abs(eth_needed) * S + abs(current_hedged_X) * S
                FREELIQs.append(newfreeliq)
                freeliq = newfreeliq
                    
            elif current_hedged_X <= 0 and eth_needed < 0: #if amm is short eth to hedge and needs to short more
                freeliq = freeliq + 2 * current_hedged_X * (S_old - S) #re-adjust how much is needed to be locked
                #print([i,eth_needed, current_hedged_X])
                lockedliq = freeliq + 2 * eth_needed * S #lock up more for shorting
                
                #print('norm')
                if lockedliq > 0: #if we have enough collateral
                    
                    #hedge_liqs.append(hedge_liqs[i] + S * abs(eth_needed) * shortRatio) #lock more collateral
                    
                    hedge_X.append([hedge_X[i][0]+eth_needed, S]) #hedge
                    
                    hedge_fees.append(S * abs(eth_needed) * hedge_cost) 
                    
                    dX.append(eth_needed)
                    
                    freeliq = lockedliq + abs(eth_needed) * S 
                    
                    FREELIQs.append(freeliq)
                    #if freeliq <0:
                        #print(freeliq)
                
                elif (lockedliq < 0) and (freeliq > 0): 
                    
                    
                    eth_needed_2 = -freeliq/(2*S) #we can't hedge too much more
                    #print([eth_needed,eth_needed_2])
                    
                    eth_hedgable = -freeliq/(2*S)
                    
                    hedge_X.append([hedge_X[i][0] + eth_needed_2, S])
                    hedge_fees.append(abs(eth_needed_2) * S *hedge_cost)
                    
                    dX.append(eth_needed_2)
                    
                    
                    
                    freeliq = freeliq - S * abs(eth_needed_2)
                    
                    
                    FREELIQs.append(freeliq)
                    #if freeliq <0:
                    #    print(freeliq)
                    #hedge_liqs.append(hedge_liqs[i]) # do nothing if no liquidity to hedge
                   # hedge_X.append([hedge_X[i][0], S])
                    #hedge_fees.append(0)
                    #dX.append(0)
                    #FREELIQs.append(freeliq)
                elif freeliq < 0:
                    
                    #hedge_liqs.append(hedge_liqs[i]) #AMM does not have enough funds to hedge
                    hedge_X.append([hedge_X[i][0], S])
                    hedge_fees.append(0)
                    dX.append(0)
                    
                    FREELIQs.append(freeliq)
                
            elif current_hedged_X <= 0 and eth_needed > 0 and current_hedged_X + eth_needed < 0:
                
                #
                #freeliq = freeliq + 2 * (current_hedged_X) * (S_old - S)
                freeliq = freeliq + abs(eth_needed) * S
                
                FREELIQs.append(freeliq)
                
                #hedge_liqs.append(hedge_liqs[i] - shortRatio * abs(eth_needed) * S + abs(eth_needed) * S) # AMM short eth, needs to long some eth to hedge but still net short
                hedge_X.append([hedge_X[i][0]+eth_needed, S])
                hedge_fees.append(S * abs(eth_needed) * hedge_cost)
                
                dX.append(eth_needed)
                
                #if freeliq <0:
                #        print(freeliq)   
                        
                        
            elif current_hedged_X <= 0 and eth_needed > 0 and current_hedged_X + eth_needed > 0:
                
                
                #hedge_liqs.append(hedge_liqs[i] - current_hedged_X * shortRatio * S + (current_hedged_X + eth_needed) * S)
                hedge_X.append([hedge_X[i][0]+eth_needed, S]) 
                hedge_fees.append(S * (eth_needed + abs(current_hedged_X)) * hedge_cost)
                dX.append(eth_needed)
                newfreeliq = freeliq - abs(eth_needed) * S + 2 * abs(current_hedged_X)*S
                FREELIQs.append(newfreeliq)
                freeliq = newfreeliq
        if orig_len  == len(hedge_X):
            hedge_X.append(hedge_X[-1]) 
            
             
                
                    
    hedge_pnl = [0]
    exp_max = np.sort(df.expiry.unique())[-1]
    final_S = final_spots[asset][str(exp_max)]
    
    
    
    #for i in range(0,len(dX)-1):
    #    long_or_shorted = dX[i+1]
       # print(error_dt[i])
        #current_spot = hedge_X[i][1]
    #    if error_dt[i] > 0.25:
    #        USE_SPOT = (df['price'].iloc[i] + df['price'].iloc[i-1])/2
           
    #    elif error_dt[i] <= .25:
     #       USE_SPOT = df['price'].iloc[i]
        #USE_SPOT = 100
    #    hedging_cost_add = long_or_shorted * USE_SPOT#df['price'].iloc[i]
        
    #    hedge_pnl.append(hedge_pnl[i] - hedging_cost_add) 
    #print(len(hedge_X))    
    for i in range(0,len(dX)-1):
        long_or_shorted = dX[i+1]
       # print(error_dt[i])
        #current_spot = hedge_X[i][1]
        if error_dt[i] > 0.25:
            USE_SPOT = (df['price'].iloc[i] + df['price'].iloc[i-1])/2
           
        elif error_dt[i] <= .25:
            USE_SPOT = df['price'].iloc[i]
        #USE_SPOT = 100
        hedging_cost_add = long_or_shorted * USE_SPOT#df['price'].iloc[i]
        
        hedge_pnl.append(hedge_pnl[i] - hedging_cost_add) 

    final_pos = hedge_X[-1][0]
    final_pnl = final_pos * final_S


    
    total_hedging_pnl = hedge_pnl[-1] + final_pnl;
    hedged_asset = [ass[0] for ass in hedge_X]  

    no_hedge_delta = -df['netDelta'] + df_collat['BaseCollateral'] + np.array(hedged_asset[1:])
    t0 = df['timestamp'].iloc[0]
    dts = (df['timestamp'] - t0)/3600
    data_frame = {"times": dts,
                  "netDelta_no_collat_no_hedge":-df['netDelta'],
                "netDelta_no_hedge": -df['netDelta'] + df_collat['BaseCollateral'],
                "hedge_eth":  np.array(hedged_asset[1:]),
                  "hedgedDelta": -df['netDelta'] + df_collat['BaseCollateral'] + np.array(hedged_asset[1:]),
                  'dX': dX[1:],
                  'FREELIQ':FREELIQs,
                  'hedgeCost': hedge_pnl[1:],
                  "spots": df['price'],
                  'HEDGEX': hedge_X[1:],
                  'SNXhedgefees':hedge_fees[1:]
                }
    return total_hedging_pnl, pd.DataFrame(data_frame), error_dt, HEDGE_TIMES
                        