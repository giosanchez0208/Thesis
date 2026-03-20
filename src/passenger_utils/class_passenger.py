from class_passenger_map import PassengerMap
import os

class Passenger:
    def __init__(self, csv_path=None):
        if csv_path is None:
            # Default path relative to this file
            csv_path = os.path.join(os.path.dirname(__file__), '../../data/processed/nodes_with_tomtom_data_imputed.csv')
        
        self.w = PassengerMap(csv_path)
        
        # Sample start_pos
        start_sample = self.w.generate_nodes(n_points=1)
        self.start_pos = start_sample['base_osmid'].iloc[0]
        
        # Sample end_pos, ensure it's different from start_pos
        while True:
            end_sample = self.w.generate_nodes(n_points=1)
            self.end_pos = end_sample['base_osmid'].iloc[0]
            if self.start_pos != self.end_pos:
                break
        
        self.curr_pos = self.start_pos
        self.total_dt = 0
        
        # method for getting lat/lon of start and end positions
        self.start_lat = start_sample['lat'].iloc[0]
        self.start_lon = start_sample['lon'].iloc[0]
        self.end_lat = end_sample['lat'].iloc[0]
        self.end_lon = end_sample['lon'].iloc[0]
        
    def get_start_lat_lon(self):
        return float(self.start_lat), float(self.start_lon)
    
    def get_end_lat_lon(self):
        return float(self.end_lat), float(self.end_lon)

if __name__ == "__main__":
    # Example usage
    passenger = Passenger()
    print(f"Passenger start position: {passenger.start_pos}")
    print(f"Passenger end position: {passenger.end_pos}")
    print(f"Passenger current position: {passenger.curr_pos}")
    print(f"Total dt: {passenger.total_dt}")
    
    print("="*20)
    print(f"Start lat/lon: {passenger.get_start_lat_lon()}")
    print(f"End lat/lon: {passenger.get_end_lat_lon()}")