import pandas as pd
import numpy as np
import os

class PassengerMap:
    def __init__(self, csv_path=None):
        if csv_path is None:
            csv_path = os.path.join(
                os.path.dirname(__file__), 
                '../../data/processed/nodes_with_tomtom_data_imputed.csv'
            )
            
        self.df = pd.read_csv(csv_path, dtype={
            'base_osmid': 'int64',
            'lat': 'float64',
            'lon': 'float64',
            'bc': 'float64',
            'bldg_density': 'float64',
            'traffic_index': 'float64',
            'ADT_prop': 'float64'
        })
        
        # Model coefficients
        self.betas = {
            'beta_0': 0.5,   # Intercept
            'beta_1': 0.6,   # ln(ADT_prop) - Traffic Intensity 
            'beta_2': 0.3,   # ln(D_bldg) - Building Density 
            'beta_3': 0.2,   # ln(C_B) - Betweenness Centrality 
            'beta_4': 0.1,   # L_mix - Land-use mix (optional) 
            'epsilon': 0.05  # Error term 
        }

    def calculate_v_ped(self, df=None):

        if df is None:
            df = self.df
        # We use a small epsilon (1e-6) to avoid errors with log(0)
        ln_v_ped = (
            self.betas['beta_0'] +
            self.betas['beta_1'] * np.log(df['ADT_prop'] + 1e-6) +
            self.betas['beta_2'] * np.log(df['bldg_density'] + 1e-6) +
            self.betas['beta_3'] * np.log(df['bc'] + 1e-6) +
            self.betas['epsilon']
        )
        # Return exponentiated result to get actual volume 
        return np.exp(ln_v_ped)

    def generate_nodes(self, n_points=10000):
 
        # Calculate V_ped for the dataframe
        self.df['v_ped'] = self.calculate_v_ped(self.df)
        
        # Normalize V_ped values to create a spatial probability distribution 
        weights = self.df['v_ped'] / self.df['v_ped'].sum()
        
        # Sample origins based on the distribution 
        sampled_indices = np.random.choice(self.df.index, size=n_points, p=weights)
        
        return self.df.loc[sampled_indices].copy()