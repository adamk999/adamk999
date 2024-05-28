import numpy as np
import pandas as pd
import torch
import requests as rq
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dt
import time
import pickle
import numpy as np
from numba import jit, int32, float32, types    # import the types
from numba.experimental import jitclass



class Misc:
    def load_pickle(self, file_name):
        # Importing the dictionary back from the pickle file
        with open(file_name, 'rb') as file:
            loaded_dict = pickle.load(file)
            
        return loaded_dict

    #spec = [
    #    ('pnl_total', float32), # a simple scalar field
    #    ("max_assets", int32),
    #    ("cash", float32),
    #    ("PnL_1", float32),
    #    ("PnL_7", float32),
    #    ("PnL_30", float32)
    #    ("strategy_map", dict)
    #]

@jitclass(spec=spec)
class Trading_Env(Misc):
    def __init__(self, episode):
        # Load data.
        self.pnl_total = 0
        self.max_assets = 4
        self.cash = 1000
        #self.drawdown = []
        self.PnL_1, self.PnL_7, self.PnL_30 = 0, 0, 0
        strategies = ["call", "put", "butterfly", "straddle", "strangle", "risk_rev", "bull_spread", "bear_spread"]
        self.strategy_map = {"call_long": {""}}
        self.positions = {strat: np.array([[]]) for strat in strategies}
        self.pos_size = 0.0 # POSITION SIZE FUNCTION OF PnL??? 
        self.pricing_history = {strat: [] for strat in strategies}

        self.S_data = self.load_pickle("Data/s_data_dict_v2.pkl")[episode]
        self.S = self.S_data["data"].drop_duplicates()
        self.S = self.S.sort_index().astype({col: float for col in self.S.columns})

        self.stock_vars = ["s_v", "s_vw", "s_o", "s_c", "s_h", "s_l", "s_n", "s_t_until_exp"]
        self.c_data = self.load_pickle("Data/call_data_dict_v2.pkl")
        self.p_data = self.load_pickle("Data/put_data_dict_v2.pkl") 

        self.expiry = self.S_data["exp_date"]
        self.date = self.S.index[0]
        self.time = self.ID = 0

        self.PnL_history = self.port_val_hist = np.zeros(shape=len(self.S))
        self.port_val_hist[0] = self.cash
        self.state_vars_hist = []


        self.type_map = {"call": {"size": 1, "label": ["c"], "direc": [1]},
                         "put": {"size": 1, "label": ["p"], "direc": [1]}, 
                         "butterfly": {"size": 4, "label": ["c"]*4, "direc": [1, -1, -1, 1]},
                         "straddle": {"size": 2, "label": ["p", "c"], "direc": [1, 1]},
                         "strangle": {"size": 2, "label": ["p", "c"], "direc": [1, 1]},
                         "risk_rev": {"size": 2, "label": ["p", "c"], "direc": [-1, 1]},
                         "bull_spread": {"size": 2, "label": ["c", "c"], "direc": [1, -1]},
                         "bear_spread": {"size": 2, "label": ["p", "p"], "direc": [-1, 1]}}


    def get_strike(self, strike):
        return [self.S[K].iloc[self.time] for K in strike]

    def create_key(self, strike):
        return f"{int(self.expiry)}_{strike}"

    def step(self, action: str, direction: int, strike: str):
        """
        Function which takes input from the RL agent and performs necessary trading logic.

        Parameters:
        action (int): Integer representing the type of order to be placed.
        direction (int): +1 meaning long a position, -1 meaning short.
        strikes (int): Relative strike price for order submission.

        Returns:
        reward (float): 
        new_price
        """

        #print(f"---------- Time: {self.time} ----------- ")

        if action != "wait": 
            self.new_pos(action, direction, strike)

        
        #for pos in self.positions.items():
         #   print(pos[0])
          #  print(pos[1])

        reward = self.portfolio_eval()

        state_vars = self.calc_analytics()
        #print("PnL_1", self.PnL_1)
        #print("Cash", self.cash)


        self.time += 1
        self.date = self.S.index[self.time]

        return [state_vars, self.PnL_1] 

    
    def new_pos(self, pos, direc, strike):
        """
        Creates new position in memory and calculates changes in cash.

        Params:
          pos: "call"
        
        """

        # Butteryfly env.step(action="butterfly", direction=1, strike=["p_1","c_1","c_2"]).
        @jit
        def position_template(pos, strike, direc):
            """
            pos == trade_type/call, put, etc
            direc = 1 (long) or -1 (short)
            """
            expiry = self.expiry
            strike_indices = [1+i for i in range(len(strike))] + [-1] # target array indices for the positions arr.
            
            positions = self.positions[pos]
            K = self.get_strike(strike)

            if direc == 1:
                # Purchasing an asset/strategy.
                if self.positions[pos].size == 0:
                    # Empty inventory.
                    self.positions[pos] = np.array([
                        [self.time, *[self.S[K].iloc[self.time] for K in strike], self.ID, direc]
                    ], dtype=float)
                    self.trade(pos, direc, self.positions[pos][0,:])
                
                elif K+[-1] in positions[:, strike_indices].tolist():
                    # Holding short asset --> buy it back.
                    self.trade(trade_type=pos, direction=direc, pos_data=self.positions[pos][0])
                    self.positions[pos] = self.positions[pos][1:] 

                else:
                    # No matched shorts --> go long and add to portfolio.
                    self.positions[pos] = np.vstack([
                        self.positions[pos],
                        np.array([self.time, *[self.S[K].iloc[self.time] for K in strike], self.ID, direc], dtype=float)
                    ])
                    self.trade(pos, direc, self.positions[pos][-1,:])

            else:
                # Selling an asset/strategy.
                if self.positions[pos].size == 0:
                    # Empty inventory.
                    self.positions[pos] = np.array([
                        [self.time, *[self.S[K].iloc[self.time] for K in strike], self.ID, direc]
                    ], dtype=float)
                    self.trade(pos, direc, self.positions[pos][0,:])


                elif K+[1] in positions[:, strike_indices].tolist():
                    # Holding assets long --> sell oldest in inventory.
                    self.trade(trade_type=pos, direction=direc, pos_data=self.positions[pos][0])
                    self.positions[pos] = self.positions[pos][1:] 


                else:
                    # No assets in inventory --> short strategy.
                    self.positions[pos] = np.vstack([
                        self.positions[pos], 
                        np.array([self.time, *[self.S[K].iloc[self.time] for K in strike], self.ID, direc], dtype=float)
                    ]) 
                    self.trade(pos, direc, self.positions[pos][-1,:])

            self.ID += 1

        # use type map to figure out whether to pass strike as list.

        if self.type_map[pos]["size"] == 1:
            position_template(pos, [strike], direc)
        else:
            position_template(pos, strike, direc)
  

    
    def trade(self, trade_type, direction, pos_data):
        """
        Take in position and calculate the change in cash.

        Params:
           pos_data: [self.time, *[self.S[K].iloc[self.time] for K in strike], expiry, direc]
           direction: 1 or -1
        """
        n_strikes = self.type_map[trade_type]["size"]
        direc = direction*np.array(self.type_map[trade_type]["direc"])

        # iterate over strikes @ time --> prices --> decrement cash.
        for i, strike in enumerate(pos_data[1:1+n_strikes]):
            key = self.create_key(strike)
            if self.type_map[trade_type]["label"][i] == "c":
                self.cash = self.cash - direc[i]*self.c_data[key]["data"].loc[self.date]["c"]
            else:
                self.cash = self.cash - direc[i]*self.p_data[key]["data"].loc[self.date]["c"]
    

    def value_asset(self, asset, asset_type, time):
        asset_value = 0
        asset_info = self.type_map[asset_type]
        directions = asset[-1]*np.array(asset_info["direc"])
        keys = [self.create_key(strike=asset[1+i]) for i in range(asset_info["size"])]
        
        for direc, key, label in zip(directions, keys, asset_info["label"]):
            if label == "c":
                asset_value += direc*self.c_data[key]["data"].loc[self.date]["c"]
            else:
                asset_value += direc*self.p_data[key]["data"].loc[self.date]["c"]

        return asset_value


    def portfolio_eval(self):
        # self.cash has to be included.
        # [self.time, *[self.S[K].iloc[self.time] for K in strike], expiry, direc]
        # Need to value everything in my portfolio.
        # Note that short assets are liabilities and must detract from the PnL. 
        
        total_value = 0
        for asset_type in list(self.positions.keys()):
            n_assets = self.positions[asset_type].size
            if n_assets != 0:
                asset_value = 0
                for asset in self.positions[asset_type]: 
                    asset_value += self.value_asset(asset, asset_type, self.time)
                   
                self.pricing_history[asset_type].append([asset_value, asset[-2], self.date])
                total_value += asset_value
        
        #print("assets", total_value, "cash", self.cash, "total", total_value + self.cash)

        self.port_val_hist[self.time] = total_value + self.cash

        if self.time > 29:
            self.PnL_30 = self.port_val_hist[self.time] - self.port_val_hist[self.time-30]
            self.PnL_7 = self.port_val_hist[self.time] - self.port_val_hist[self.time-7] 
            self.PnL_1 = self.port_val_hist[self.time] - self.port_val_hist[self.time-1] 

        elif self.time > 6:
            self.PnL_7 = self.port_val_hist[self.time] - self.port_val_hist[self.time-7]
            self.PnL_1 = self.port_val_hist[self.time] - self.port_val_hist[self.time-1] 

        elif self.time > 0:
            self.PnL_1 = self.port_val_hist[self.time] - self.port_val_hist[self.time-1]
    

        return total_value
                
    def calc_analytics(self):
        """——> Should be a focus on relative changes and distances.
        Precalculated:
        - Price of different assets (stock/call/put) --> s_vars + pct_x_y. DONE
        - %change price (diff horizons) --> 1 step, 7 step, 30 step, 100 step. DONE
        - Time until expiration (same for each) --> t until expiration. DONE
        - Distance to significant price level (rounded to 10) --> below + above. DONE
        - AutoARIMA estimate for the next close price (res_calc every 15 steps, win_size=100).
        - LSTM estimate for the next WVAP. DONE
        - Distance to the 7, 30 period MA. DONE
        
        Calc here.
        - PnL tracking --> 1 step, 7 step, 30 step.
        - PnL per strategy --> calc here.                                         
        - PnL for oldest long/short on each strategy (for deletion) --> calc here.        

        Unsure:
        - Current vol/implied vol.
        - Black Scholes price.
        - Greeks.
        """
        state_vars = []
        
        analytics = ["dist_ab", "dist_bel", "7_ma_dist", "30_ma_dist", "100_ma_dist"]
        pct_changes = [f"pct_s_{t}" for t in [1, 10, 50]] + [f"pct_{pc}{i}" for pc in ["c", "p"] for i in range(3)]
        
        [state_vars.append(self.S[item].iloc[self.time]) for item in analytics + pct_changes]
        [state_vars.append(PnL) for PnL in [self.PnL_1, self.PnL_7, self.PnL_30]]
        state_vars.append(self.S["s_t_until_exp"].iloc[self.time])

        self.state_vars_hist.append(state_vars)
        
        return torch.Tensor(state_vars)

    def reset(self):
        return self.calc_analytics(), len(self.S) - 1


start = dt.datetime.now()

env = Trading_Env(episode=1)

env.step(action="call", direction=-1, strike="c_1")
env.step(action="call", direction=1, strike="c_1")


for i in range(40):
   env.step(action="butterfly", direction=1, strike=["c_0","c_1","c_1", "c_0"])
   env.step(action="call", direction=-1, strike="c_1")

print(dt.datetime.now() - start)


