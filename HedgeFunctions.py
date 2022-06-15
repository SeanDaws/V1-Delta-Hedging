import pandas as pd
import numpy as np
import datetime, time
from copy import deepcopy
import numpy as np
from pandas import read_excel
from time import sleep
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from Delta_Cutoffs import *

SNX_fee = 0.001;

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
 '1647590400': 2795.09,
 '1646985600': 2585.105,
 '1648195200': 3132.6333}

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
 '1648195200': 16.043429}

#final_spots_SOL = {'1646380800':91.93384197,
#                  '1647590400': 85.746802, '1648195200', 102.99, }

final_spots_SOL = {'1646380800': 91.93384197, 
                   '1646985600': 82.29, 
                   '1647590400': 85.746802, 
                   '1648195200':102.99,
                   '1648800000':126.4}
                   #'1649404800': 119.04054787, 
                   #'1650009600': 101.71586655, 
                    


def true_cost(x):
    vol = x['baseIv'] * x['skew']
    K = x['strike']
    S = x['price']
    rate = 0.08
    T = x['tau']/(60*60*24*365)
    paid_cost = x['totalCost']
    
    if x['OpenClose'] == 'Open':
        sign = +1
    else:
        sign = -1
    if (x['tradeType'] == 'LONG_CALL'):
        trueCost = x['amount'] * bs_call(vol, K, S, rate, T)
        edge = sign*(paid_cost - trueCost) #- x['amount'] *0.003*S
    elif (x['tradeType'] == 'SHORT_CALL'):
        trueCost = x['amount'] *bs_call(vol, K, S, rate, T)
        edge = sign * (trueCost - paid_cost) #- x['amount'] *0.003*S
    elif (x['tradeType'] == 'LONG_PUT'):
        trueCost = x['amount'] * bs_put(vol, K, S, rate, T)
        edge = sign * (paid_cost - trueCost) #- x['amount'] *0.003*S
    elif (x['tradeType'] == 'SHORT_PUT'):
        trueCost = x['amount'] *bs_put(vol, K, S, rate, T)
        edge = sign * (trueCost - paid_cost) #- x['amount'] *0.003*S
        
    return edge


def flip_cost(x):
    if x['OpenClose'] == 'Open':
        sign = +1
    else: 
        sign = -1
    
    if x['tradeType'] == 'LONG_CALL':
        return sign * x['totalCost']
    elif x['tradeType'] == 'SHORT_CALL':
        return -sign * x['totalCost']
    elif x['tradeType'] == 'LONG_PUT':
        return sign * x['totalCost']
    elif x['tradeType'] == 'SHORT_PUT':
        return -sign * x['totalCost']
    
    
def flip_amount(x):
    if x['OpenClose'] == 'Open':
        sign = +1
    else: 
        sign = -1
    
    if x['tradeType'] == 'LONG_CALL':
        return sign * (-x['amount'])
    elif x['tradeType'] == 'SHORT_CALL':
        return sign * x['amount']
    elif x['tradeType'] == 'LONG_PUT':
        return sign * (-x['amount'])
    elif x['tradeType'] == 'SHORT_PUT':
        return sign * x['amount']
    
def option_value_at_exp(x, FINAL_SPOTS):
    exp = x['expiry']
    
    S = FINAL_SPOTS[str(exp)]
    
    if x['tradeType'] == "CALL":
        pnl = x['amount'] * max(S- x['strike'], 0)
    elif x['tradeType'] == 'PUT':
        pnl = x['amount'] * max(x['strike'] - S, 0)
    return pnl

def callput(x):
    if (x['tradeType'] == "LONG_CALL") | (x['tradeType'] == "SHORT_CALL"):
        return "CALL"
    elif (x['tradeType'] == "LONG_PUT") | (x['tradeType'] == "SHORT_PUT"): 
        return "PUT"
    
def Pos_Generator(df, t):
    exps = np.sort(df.expiry.unique())
    last_exp = 0;
    
    
    old_exps = exps[exps<t]
        
    if len(old_exps) ==0:
        last_exp = 0

    else:
        last_exp = min(old_exps, key=lambda x:abs(x-t))
        
    look_back = t - 60*60*24*7*4.1
    
    df_copy = df.copy()
    df_copy = df_copy.loc[(df['timestamp'] > look_back) & (df['timestamp'] <=  t) & (df['expiry'] > last_exp)]
    #df = df.loc[df['expiry'] >= t]
    
    return df_copy #Fi


def insert_collat1(x):
    if (x['tradeType'] == "LONG_CALL") & (x['OpenClose'] == "Open"):
        return x['amount']
    elif (x['tradeType'] == "LONG_CALL") & (x['OpenClose'] == "Close"):
        return -x['amount']
    else:
        return 0
    
def insert_collat2(x):
    if (x['tradeType'] == "LONG_CALL") & (x['OpenClose'] == "Open"):
        return -x['amount'] * x['price'] - x['price']* x['amount'] * SNX_fee
    elif (x['tradeType'] == "LONG_CALL") & (x['OpenClose'] == "Close"):
        return x['amount'] * x['price'] - x['price']* x['amount'] * SNX_fee 
    else:
        return 0
    
    
def add_collateral_delta(df):
    df2 = df.copy()
    LEN = len(df2)
    zeros = [0 for i in range(LEN)]
    df2['ETH collat'] = zeros
    df2['cost to collat'] = zeros
    
    df2['ETH collat'] = df2.apply(lambda x: insert_collat1(x), axis = 1)
    df2['cost to collat']= df2.apply(lambda x: insert_collat2(x), axis = 1)
    #zip(*df2.apply(lambda x: insert_collat(x), axis = 1))
    return df2

    
def round_df(df, rounds, which_round):
    start_index = rounds[which_round]
    end_index = rounds[which_round +1]
    
    return  df.iloc[start_index:end_index]


def cum_add(x):
    if (x['tradeType'] == "LONG_CALL") & (x['OpenClose'] == "Open"):
        return x['amount'], x['amount'] * x['price']
    elif (x['tradeType'] == "LONG_CALL") & (x['OpenClose'] == "Close"):
        return -x['amount'], -x['amount'] * x['price']

def tack_on_collat(df): 
    new_df = add_collateral_delta(df)
    ZERO = [0 for i in range(0, len(new_df))]
    new_df['netDelta'] = -new_df['netDelta']
    
    exp_dict = {};
    for exp in new_df.expiry.unique():

        exp_dict['{}'.format(exp)] = [0 for i in range(0, len(new_df))]
        
  
    for i in range(0,len(new_df)):
        x = new_df.iloc[i]
        EXP = x['expiry']

        if (x['tradeType'] == "LONG_CALL") & (x['OpenClose'] == "Open"):
            exp_dict[str(EXP)][i] = x['amount']
            
            
        elif (x['tradeType'] == "LONG_CALL") & (x['OpenClose'] == "Close"):
            exp_dict[str(EXP)][i] = -x['amount']
            
            
    for key in exp_dict:
        new_df[str(key)] = exp_dict[key]
    
    for exp in new_df.expiry.unique():
        new_df['cum sum {}'.format(exp)] = new_df[str(exp)].cumsum()
        
    return new_df


def Hedger(freq, which_round, rounds, starting_sUSD, df, FINAL_SPOTS, HEDGE_SHIFT = 0):
    interest_rate = 0.10; # this is the interest rate paid on shorts
    start_index = rounds[which_round]
    end_index = rounds[which_round +1] #Find the index for the start and end of the round
    
    sUSD = starting_sUSD[which_round] #find how much sUSD there is for the round 
    hedge_liq = 0.33 * sUSD #find how much liquidity there is to hedge
    
    
    
    df_mini = df.iloc[start_index:end_index]
    
    max_exp = max(df_mini.expiry.unique())
    
    FINAL_SPOT = FINAL_SPOTS[str(max_exp)]
    
    Timestamps = df.iloc[start_index:end_index]['timestamp'].to_numpy()
    Spots = df[start_index:end_index]['price'].to_numpy()
    netDeltas = df[start_index:end_index]['netDelta'].to_numpy() #Find the times, spots and netdeltas for the round

    t0 = Timestamps[0]
    hedge_time = t0 + freq * 60*60 + HEDGE_SHIFT *60 * 60; #initialise the first hedge time
    
    ETH = 0.00001; # initialise the pool's initial sETH position
    
    True_Net_Delta = []; # initialise arrays to store the pool's "true" net delta, sETH position and hedging fees
    ETH_series = [];
    
    hedging_Fees = [];
    
    liqs = [];
    
    unique_exps = df.expiry.unique();
    
    netDelta_with_collat = []

    tot_delta = []
    
    short_rate_paid = [];
    
    long_short_eth = [];
    
    for i in range(0, len(Timestamps)):
        current_time = Timestamps[i]
        actual_spot = Spots[i]
        current_spot = np.mean(Spots)
        
        ETH_collat = 0
        
        for exp in unique_exps:
            if current_time < exp:
                to_add = df_mini.iloc[i]['cum sum {}'.format(exp)]
                #print(df_mini.iloc[i]['cum sum {}'.format(exp)])
                ETH_collat += to_add
        
        
        total_delta = netDeltas[i] + ETH + ETH_collat
        
        rate_paid = 0
        hedge_fee = 0;
        
        
        
        if (i>0) & (ETH < 0): #if the pool is short sETH for hedging, add the fee paid to short
            dt = current_time - Timestamps[i-1];
            
            rate_paid = ETH * dt*interest_rate/(60*60*24*365)   
            ETH -= rate_paid
            short_rate_paid.append(rate_paid)
            
            
        #begin hedge
        if current_time > hedge_time: #Each time we hit a hedge time:
            
            eth_needed = -total_delta #find how much ETH is needed to hedge
            
            if (np.sign(eth_needed) < 0) & (np.sign(ETH) <0): #if pool is short sETH and needs to short more
                
                hedge_liq_needed = 2 * abs(ETH + eth_needed) * current_spot #find how much liquidity is needed to hedge
                
                
                if hedge_liq_needed <= hedge_liq: 
                    
                    ETH += eth_needed
                    hedge_fee = abs(eth_needed) * actual_spot* SNX_fee
                    hedge_liq -= hedge_fee

                    long_short_eth.append([eth_needed,actual_spot])
                    
                     
            elif (np.sign(eth_needed) >0) & (np.sign(ETH) >0): #if pool is long sETH and needs to long more
                
                hedge_liq_needed = abs(ETH + eth_needed) * current_spot #find how much liquidity is needed to hedge

                if hedge_liq_needed <= hedge_liq:           
                    ETH += eth_needed
                    hedge_fee = eth_needed * actual_spot * SNX_fee
                    hedge_liq -= hedge_fee
                    long_short_eth.append([eth_needed,actual_spot])
                    
            
            elif (np.sign(eth_needed) <0) & (np.sign(ETH) >0): # if pool is long sETH and needs to short sETH
                
                if ETH + eth_needed > 0: #if pool will still be long sETH, just close some longs
                    ETH += eth_needed
                    
                    hedge_fee = abs(eth_needed) * actual_spot * SNX_fee
                    hedge_liq -= hedge_fee
                    long_short_eth.append([eth_needed,actual_spot])
                   
                elif ETH + eth_needed < 0: # if pool will end up net short, close to 0, then go short
                    
                    hedge_fee = abs(eth_needed) * actual_spot* SNX_fee
                    
                    hedge_liq -= hedge_fee
                    
                    ETH = eth_needed + ETH;
                    
                    long_short_eth.append([eth_needed,actual_spot])
                    
            elif (np.sign(eth_needed) >0) & (np.sign(ETH) <0): # if pool is short sETH and needs to long sETH
                if ETH + eth_needed < 0: #if the pool will keep being short
                    ETH += eth_needed # just close the shorts
                    hedge_fee = abs(eth_needed) * actual_spot* SNX_fee
                    hedge_liq -= hedge_fee
                    long_short_eth.append([eth_needed,actual_spot]) 
                    
                elif ETH + eth_needed > 0: #pool will change to being long sETH
                    hedge_fee = abs(eth_needed) * actual_spot*SNX_fee
                    
                    ETH = ETH + eth_needed
                                 
                    hedge_liq -= hedge_fee
                    long_short_eth.append([eth_needed,actual_spot])
                        
            total_delta = netDeltas[i] + ETH + ETH_collat

            hedge_time += freq * 60*60      
            

        True_Net_Delta.append(total_delta)
        
        total_fee = abs(hedge_fee)
       
        ETH_series.append(ETH)
        
        hedging_Fees.append(total_fee)
        liqs.append(hedge_liq)
        netDelta_with_collat.append(netDeltas[i] + ETH_collat)
        
    
    short_rate_fee = np.sum(short_rate_paid)*current_spot
    
    eth_pos = 0
            
    for hedge in long_short_eth:
        eth_pos += - hedge[0] * hedge[1]
        
    final_ETH_pos_val = ETH * FINAL_SPOT#are these signs correct?
     
    long_short_pnl = eth_pos + final_ETH_pos_val
    
    final_fee = final_ETH_pos_val* SNX_fee
    
    hedging_Fees.append(abs(final_fee))

    return Timestamps, netDelta_with_collat, ETH_series, True_Net_Delta, hedging_Fees, liqs, Spots,long_short_pnl, short_rate_fee

def HedgeVisualize(freqs, which_round, rounds,starting_sUSD, df, plotornot, FINAL_SPOTS, HEDGE_SHIFT):
    
    exps = df.expiry.unique()

    ts, UHDeltas, ETHs, HedgeDeltas, fees, liqs, spots, hedge_PNL, short_rate_fee = Hedger(freqs, which_round, rounds, starting_sUSD,df,FINAL_SPOTS, HEDGE_SHIFT)
    
    
    start_index = rounds[which_round]
    end_index = rounds[which_round +1] #Find the index for the start and end of the round
    
    df_mini = df.iloc[start_index:end_index]
    
    
    exps = df.expiry.unique()
    start_time = ts[0]
    end_time = ts[-1]
        
    
    if plotornot == "plot":
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3,sharex=True, figsize=(12,5))
        valid_exps = exps[(exps > start_time) & (exps < end_time)]
    
        ax1.plot(ts, HedgeDeltas)
        ax1.plot(ts, UHDeltas)
        ax1.plot(ts,df_mini['netDelta'])
    
        ax2.plot(ts, ETHs)
        for exp in valid_exps:
            ax1.axvline(x=exp, color = 'red', linestyle='--')
            ax2.axvline(x=exp, color = 'red', linestyle='--')
            
        ax1.set_xlabel("time")
        ax2.set_xlabel('time')
        ax1.set_ylabel("NetDelta")
        ax2.set_ylabel("sETH")
        
        ax3.plot(ts,liqs)
        
    return fees, ETHs, hedge_PNL, short_rate_fee


def flips(df):
    df2 = df.copy()
    df2['totalCost'] = df2.apply(lambda x: flip_cost(x), axis=1)
    df2['amount'] = df2.apply(lambda x: flip_amount(x), axis =1)
    df2['tradeType'] = df2.apply(lambda x: callput(x), axis =1)
    return df2

def premiums_received(df, rounds, which_round):
    start_index = rounds[which_round]
    end_index = rounds[which_round +1]
    
    df_round = df.iloc[start_index:end_index]
    
    df_round2 = flips(df_round)
    
    premium_received = df_round2['totalCost'].sum()
    return premium_received


def Options_at_Exp(df, rounds, which_round, FINAL_SPOTS):
    start_index = rounds[which_round]
    end_index = rounds[which_round +1]
    
    df_round = df.iloc[start_index:end_index]
    df_round2 = flips(df_round)
    df_round3 = df_round2.groupby(['expiry','strike','tradeType'])['amount'].sum().reset_index()
    
    df_round3['OptionValue'] = df_round3.apply(lambda x: option_value_at_exp(x, FINAL_SPOTS), axis = 1)  
    
    return df_round3['OptionValue'].sum()


def collateral_PNL(df, rounds, which_round, FINAL_SPOTS):
    start_index = rounds[which_round]
    end_index = rounds[which_round +1]
    
    df_round = df.iloc[start_index:end_index]
    
    df_round2 = tack_on_collat(df_round).reset_index()
        
    collat_tot = 0;
    
    for exp in df_round2.expiry.unique():
        
        n_collat_at_exp = df_round2.iloc[-1]['cum sum {}'.format(str(exp))]
        
        price_at_exp = FINAL_SPOTS[str(exp)]
        collat_tot += n_collat_at_exp * price_at_exp
    
    collat_paid = df_round2['cost to collat'].sum()
    collat_received = collat_tot
    
    #return collat_received - collat_paid NOT SURE IF THIIS IS CORRECT
    return collat_received + collat_paid


def PNL_total(df, round_starts, which_round, FINAL_SPOTS):
    prem = premiums_received(df, round_starts, which_round)
    opt_val = Options_at_Exp(df, round_starts, which_round, FINAL_SPOTS)
    collat_pnl = collateral_PNL(df, round_starts, which_round, FINAL_SPOTS)
    
    return prem, opt_val, collat_pnl



def find_spot(df, t):
    closest_index = find_nearest(df['timestamp'], t)


    if df.iloc[closest_index]['timestamp'] <t:
        other_index = closest_index + 1
    elif df.iloc[closest_index]['timestamp'] >t:
        other_index = closest_index - 1
        
    spot_av = (df.iloc[closest_index]['price'] + df.iloc[other_index]['price'])/2.0
    return spot_av

def Checker(df):
    lens = [];
    for t in df['timestamp']:
        lens.append(len(Pos_Generator(df, t)))
    return(lens)


def FeeFinder(x):
    vol = x['baseIv'] * x['skew']
    K = x['strike']
    S = x['price']
    rate = 0.06
    T = x['tau']/(60*60*24*365)
    paid_cost = x['totalCost']
    
    
    spotFee = x['amount'] * SNX_fee * S
    
    if x['OpenClose'] == 'Open':
        sign = +1
    else:
        sign = -1
    
    if (x['tradeType'] == 'LONG_CALL'):
        trueCost = x['amount'] * bs_call(vol, K, S, rate, T)
        optionFee = trueCost*.01
        VuFee = sign * (x['totalCost'] - trueCost) - optionFee - spotFee
    
    elif (x['tradeType'] == 'SHORT_CALL'):
        trueCost = x['amount'] *bs_call(vol, K, S, rate, T)
        optionFee = trueCost*.01
        VuFee = sign * (trueCost - x['totalCost']) - optionFee - spotFee
    
    elif (x['tradeType'] == 'LONG_PUT'):
       
        trueCost = x['amount'] * bs_put(vol, K, S, rate, T)
        optionFee = trueCost*.01
        VuFee = sign* (x['totalCost'] - trueCost) - optionFee - spotFee
    
    elif (x['tradeType'] == 'SHORT_PUT'):
        trueCost = x['amount'] *bs_put(vol, K, S, rate, T)
        optionFee = trueCost*.01
        VuFee = sign * (trueCost - x['totalCost']) - optionFee - spotFee
        
    
        
    return trueCost, spotFee, optionFee, VuFee



def AMM_Collat_Fee_Finder(df, rounds, which_round):
    start_index = rounds[which_round]
    end_index = rounds[which_round +1]
    
    df_round = df.iloc[start_index:end_index]
    
    df_round2 = tack_on_collat(df_round).reset_index()
    
    df_round2['AMM collat fee'] = df_round2.apply(lambda x: add_collat_fee_cost(x), axis = 1)
    
    return df_round2


def fee_df(df,  which_round, rounds):
    start_index = rounds[which_round]
    end_index = rounds[which_round +1]
    
    df_round = df.iloc[start_index:end_index]
    df_new = df_round.copy()
    df_new['AMM collat fee'] = df_new.apply(lambda x: add_collat_fee_cost(x), axis = 1)
    df_new[['TrueCost','SpotFee','OptionFee','VUFee']] = df_new.apply(FeeFinder, axis = 1, result_type='expand')
    return df_new


def add_collat_fee_cost(x):
    if x['tradeType'] == 'LONG_CALL':
        return x['amount'] * x['price'] * SNX_fee
    else:
        return 0
