import numpy as np
import pandas as pd
import torch
import requests as rq
import matplotlib.pyplot as plt
import datetime as dt
import time
import pickle
np.set_printoptions(suppress=True)

class Misc:
    def load_pickle(self, file_name):
        # Importing the dictionary back from the pickle file
        with open(file_name, 'rb') as file:
            loaded_dict = pickle.load(file)
            
        return loaded_dict


class Trading_Env(Misc):
    """
    The trading environment which handles all the pricing computation, portfolio logic,
    and reward calculation.
    """
    def __init__(self, episode: int):
        """
        Initialise the strategies and data for the specified episode:

        Params:
         episode (int): ranging 0-24 for the 25 options periods through which the agent can trade.
        """
        # Load data.
        self.cash = 1000 # strating balance of cash.
        self.drawdown = []  # redundant drawdown array.
        self.PnL_1, self.PnL_7, self.PnL_30 = 0, 0, 0
        strategies = ["call", "put", "butterfly", "straddle", "strangle", "risk_rev", "bull_spread", "bear_spread"]
        self.positions = {strat: np.array([[]]) for strat in strategies} # inventory.
        self.pos_size = 0.0 # POSITION SIZE FUNCTION OF PnL??? 
        self.pricing_history = {strat: [] for strat in strategies}

        self.S_data = self.load_pickle("Data/s_data_dict_v2.pkl")[episode] # only load the specified episode.
        self.S = self.S_data["data"].drop_duplicates()
        self.S = self.S.sort_index().astype({col: float for col in self.S.columns}) # force types just in case.

        self.stock_vars = ["s_v", "s_vw", "s_o", "s_c", "s_h", "s_l", "s_n", "s_t_until_exp"]
        self.c_data = self.load_pickle("Data/call_data_dict_v2.pkl") # load call data.
        self.p_data = self.load_pickle("Data/put_data_dict_v2.pkl") # load put data.

        self.expiry = self.S_data["exp_date"]
        self.date = self.S.index[0]
        self.time = self.ID = 0

        # Create tracker arrays.
        self.port_val_hist = np.zeros(shape=len(self.S))
        self.port_val_hist[0] = self.cash
        self.state_vars_hist = []
        self.asset_history = np.zeros(shape=(len(self.S), len(self.S))) 

        # Type map used by trading logic in the value_asset function. 
        self.type_map = {"call": {"size": 1, "label": ["c"], "direc": [1]},
                         "put": {"size": 1, "label": ["p"], "direc": [1]}, 
                         "butterfly": {"size": 4, "label": ["c"]*4, "direc": [1, -1, -1, 1]},
                         "straddle": {"size": 2, "label": ["p", "c"], "direc": [1, 1]},
                         "strangle": {"size": 2, "label": ["p", "c"], "direc": [1, 1]},
                         "risk_rev": {"size": 2, "label": ["p", "c"], "direc": [-1, 1]},
                         "bull_spread": {"size": 2, "label": ["c", "c"], "direc": [1, -1]},
                         "bear_spread": {"size": 2, "label": ["p", "p"], "direc": [-1, 1]}}


    def get_strike(self, strike: list[str]):
        """
        Helper function which takes in a strike label and outputs the corresponding
        numerical strike in the present period e.g. "c_1" --> 150.0.

        Params:
         strike List(str): the strike label(s), e.g. "p_1".

        Returns:
         array of strke values corresponding to input.
        """
        return [self.S[K].iloc[self.time] for K in strike]

    def create_key(self, strike: float):
        """
        Helper function to format the strike key correctly, e.g. 150.0 --> "231117_150.0".
        
        Params:
         strike (float): numerical strike value.

        Returns:
         label for use in querying options dictionary.
        """
        return f"{int(self.expiry)}_{strike}"

    def step(self, action: str, direction: int, strike: str):
        """
        Function which takes input from the RL agent and performs necessary trading logic.

        Parameters:
         action (str): Representing the type of order to be placed, e.g. "call".
         direction (int): +1 meaning long a position, -1 meaning short.
         strikes (str): Relative strike price for order submission, e.g. "c_1".

        Returns:
         reward (float): PnL over 1 period.
         state_vars (list): all the calculated state variables for a step.
        """

        #print(f"---------- Time: {self.time} ----------- ")

        if action != "wait": 
            self.new_pos(action, direction, strike)

        
        #for pos in self.positions.items():
            #print(pos[0])
            #print(pos[1])

        self.time += 1
        self.date = self.S.index[self.time]

        reward = self.portfolio_eval()

        state_vars = self.calc_analytics()
        #print("PnL_1", self.PnL_1)
        #print("Cash", self.cash)


        return [state_vars, self.PnL_1] 


    def new_pos(self, pos: str, direc: int, strike: list[str]):
        """
        Creates new position in memory and calculates changes in cash.

        Params:
          pos (str): the position type to make e.g. "call"
          direc (int): 1 = call, -1 = put.
          strike (List[str]): list of strings representing strikes to trade at.
        
        """

        # Butteryfly env.step(action="butterfly", direction=1, strike=["p_1","c_1","c_2"]).

        def position_template(pos: str, strike: list[str], direc: int):
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
                
                elif -1 in positions[:, -1].tolist():
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


                elif 1 in positions[:, -1].tolist():
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
  

    
    def trade(self, trade_type: str, direction: int, pos_data: np.array):
        """
        Take in position and calculates the change in cash in-place.

        Params:
         trade_type (str): the type of trade under consideration e.g. call. 
         pos_data (np.array): [self.time, *[self.S[K].iloc[self.time] for K in strike], expiry, direc]
         direction (int): 1 or -1
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
    

    def value_asset(self, asset: np.array, asset_type: str, time: int):
        """
        Function which values a given asset from an asset type. Can be valued at different (non-current) times.

        Params:
         asset (np.array): asset data from the portfolio.
         asset_type (str): strategies being valued.
         time (int): time at which to value.

        Returns:
         asset_value (float): value of the asset.
        """
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
        """
        Function which iterates over the inventory and estimates the value of the portfolio.
        
        Returns:
        total_value (float): self-explanatory.
        """
        
        total_value = 0
        for asset_type in list(self.positions.keys()):
            n_assets = self.positions[asset_type].size
            if n_assets != 0:
                asset_value = 0
                for asset in self.positions[asset_type]:
                    temp_val = self.value_asset(asset, asset_type, self.time) 
                    asset_value += temp_val

                    if asset_hist:
                        self.asset_history[int(asset[0]), self.time] = temp_val

                   
                self.pricing_history[asset_type].append([asset_value, asset[-2], self.time])
                total_value += asset_value
        
        #print("assets", total_value, "cash", self.cash, "total", total_value + self.cash)

        self.port_val_hist[self.time] = total_value + self.cash

        # Done this way to minimise computation given time mostly > 29; others aren't triggered.
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
                
    def calc_analytics(self) -> torch.FloatTensor:
        """
        Function which calculates state variables and returns them as a FloatTensor.

        Precalculated:
        - Price of different assets (stock/call/put) --> s_vars + pct_x_y. DONE
        - %change price (diff horizons) --> 1 step, 7 step, 30 step, 100 step. DONE
        - Time until expiration (same for each) --> t until expiration. DONE
        - Distance to significant price level (rounded to 10) --> below + above. DONE
        - AutoARIMA estimate for the next close price (res_calc every 15 steps, win_size=100).
        - LSTM estimate for the next WVAP. DONE
        - Distance to the 7, 30 period MA. DONE
        
        Calc here.
        - PnL tracking --> 1 step, 7 step, 30 steps
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
        state_vars.append(self.S["Y_hat"].iloc[self.time])
        self.state_vars_hist.append(state_vars)
        
        return torch.FloatTensor(state_vars)

    def reset(self):
        """ 
        Reset calculate the analytics and return the length of the episode.

        Returns:
         state_vars, episode_length
        """
        return self.calc_analytics(), len(self.S) - 1







