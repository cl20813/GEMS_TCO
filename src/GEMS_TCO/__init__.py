import os
import pickle
import sys
import pandas as pd
import numpy as np
import torch

from pathlib import Path
import json
from json import JSONEncoder
import csv
from typing import List, Tuple, Dict, Any

gems_tco_path = "/Users/joonwonlee/Documents/GEMS_TCO-1/src"
sys.path.append(gems_tco_path)

from GEMS_TCO import orderings as _orderings
from GEMS_TCO import configuration as config


# This line makes the class available directly from the package
from .data_loader import load_data2


class log_semivariograms:
    def __init__(self, deltas: List[Tuple[float, float]], semivariograms, tolerance: float):
        """
        Initialize the log semivariograms parameters.

        Parameters:
        - deltas (List[Tuple[float, float]]): List of deltas for semivariogram calculation.
        - tolerance (float): Tolerance level for semivariogram calculation.
        """
        self.deltas = deltas
        self.tolerance = tolerance

    def toJSON(self) -> str:
        """
        Convert the object to a JSON string.

        Returns:
        - str: JSON representation of the object.
        """
        return json.dumps(self, cls=alg_opt_Encoder, sort_keys=False)

    def save(self, input_filepath: Path, data: Any) -> None:
        """
        Save the aggregated data back to the JSON file.

        Parameters:
        - input_filepath (Path): Path to the JSON file.
        - data (Any): Data to be saved.
        """
        with input_filepath.open('w', encoding='utf-8') as json_file:
            json_file.write(json.dumps(data, separators=(",", ":"), indent=4))

    def load(self, input_filepath: Path) -> Any:
        """
        Load data from a JSON file.

        Parameters:
        - input_filepath (Path): Path to the JSON file.

        Returns:
        - Any: Loaded data.
        """
        try:
            with input_filepath.open('r', encoding='utf-8') as f:
                loaded_data = json.load(f)
        except FileNotFoundError:
            loaded_data = []
        return loaded_data
    
    def tocsv(self, jsondata: List[str], fieldnames: List[str], csv_filepath: Path) -> None:
        """
        Convert JSON data to CSV format.

        Parameters:
        - jsondata (List[str]): List of JSON strings.
        - fieldnames (List[str]): List of field names for the CSV.
        - csv_filepath (Path): Path to the CSV file.
        """
        data_dicts = [json.loads(data) for data in jsondata]
        with csv_filepath.open(mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for data in data_dicts:
                writer.writerow(data)

class alg_optimization:
    def __init__(self, day: int, cov_name: str, lat_lon_resolution: List[int], lr: float, stepsize: float, params: List[float], time: float, frob_norm: float):
        """
        Initialize the optimization algorithm parameters.

        Parameters:
        - day (int): Day of the optimization.
        - cov_name (str): Name of the covariance model.
        - lat_lon_resolution (List[int]): Resolution for latitude and longitude.
        - lr (float): Learning rate.
        - stepsize (float): Step size for the optimization.
        - params (List[float]): List of parameters for the model.
        - time (float): Time parameter.
        """
        self.day = day
        self.cov_name = cov_name
        self.lat_lon_resolution = lat_lon_resolution
        self.lr = lr
        self.stepsize = stepsize
        self.sigma = params[0]
        self.range_lat = params[1]
        self.range_lon = params[2]
        self.advec_lat = params[3]
        self.advec_lon = params[4]
        self.beta = params[5]
        self.nugget = params[6]
        self.loss = params[7]
        self.time = time
        self.frob_norm = frob_norm

    def toJSON(self) -> str:
        """
        Convert the object to a JSON string.

        Returns:
        - str: JSON representation of the object.
        """
        return json.dumps(self, cls=alg_opt_Encoder, sort_keys=False)

    def save(self, input_filepath: Path, data: Any) -> None:
        """
        Save the aggregated data back to the JSON file.

        Parameters:
        - input_filepath (Path): Path to the JSON file.
        - data (Any): Data to be saved.
        """
        with input_filepath.open('w', encoding='utf-8') as json_file:
            json_file.write(json.dumps(data, separators=(",", ":"), indent=4))

    def load(self, input_filepath: Path) -> Any:
        """
        Load data from a JSON file.

        Parameters:
        - input_filepath (Path): Path to the JSON file.

        Returns:
        - Any: Loaded data.
        """
        try:
            with input_filepath.open('r', encoding='utf-8') as f:
                loaded_data = json.load(f)
        except FileNotFoundError:
            loaded_data = []
        return loaded_data
    
    def tocsv(self, jsondata: List[str], fieldnames: List[str], csv_filepath: Path) -> None:
        """
        Convert JSON data to CSV format.

        Parameters:
        - jsondata (List[str]): List of JSON strings.
        - fieldnames (List[str]): List of field names for the CSV.
        - csv_filepath (Path): Path to the CSV file.
        """
        data_dicts = [json.loads(data) for data in jsondata]
        with csv_filepath.open(mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for data in data_dicts:
                writer.writerow(data)
                

class alg_opt_Encoder(JSONEncoder):
    """
    Custom JSON encoder for alg_optimization objects.
    """
    def default(self, o: Any) -> Dict[str, Any]:
        """
        Override the default method to handle alg_optimization objects.

        Parameters:
        - o (Any): Object to be encoded.

        Returns:
        - Dict[str, Any]: Dictionary representation of the object.
        """
        if isinstance(o, alg_optimization):
            return o.__dict__
        return super().default(o)  # delegates the serialization process to the standard JSONEncoder


class likelihood_comparison:
    def __init__(self, day: int, cov_name: str, lat_lon_resolution: List[int], params: List[float], time: float):
        """
        Initialize the likelihood comparison parameters.

        Parameters:
        - day (int): Day of the comparison.
        - cov_name (str): Name of the covariance model.
        - lat_lon_resolution (List[int]): Resolution for latitude and longitude.
        - params (List[float]): List of parameters for the model.
        - time (float): Time parameter.
        """
        self.day = day
        self.cov_name = cov_name
        self.lat_lon_resolution = lat_lon_resolution
        
        self.sigma = params[0]
        self.range_lat = params[1]
        self.range_lon = params[2]
        self.advec_lat = params[3]
        self.advec_lon = params[4]
        self.beta = params[5]
        self.nugget = params[6]
        self.loss = params[7]
        self.time = time
   
    def toJSON(self) -> str:
        """
        Convert the object to a JSON string.

        Returns:
        - str: JSON representation of the object.
        """
        return json.dumps(self, cls=likelihood_comp_Encoder, sort_keys=False)

    def save(self, input_filepath: Path, data: Any) -> None:
        """
        Save the aggregated data back to the JSON file.

        Parameters:
        - input_filepath (Path): Path to the JSON file.
        - data (Any): Data to be saved.
        """
        with input_filepath.open('w', encoding='utf-8') as json_file:
            json_file.write(json.dumps(data, separators=(",", ":"), indent=4))

    def load(self, input_filepath: Path) -> Any:
        """
        Load data from a JSON file.

        Parameters:
        - input_filepath (Path): Path to the JSON file.

        Returns:
        - Any: Loaded data.
        """
        try:
            with input_filepath.open('r', encoding='utf-8') as f:
                loaded_data = json.load(f)
        except FileNotFoundError:
            loaded_data = []
        return loaded_data
    
    def tocsv(self, jsondata: List[str], fieldnames: List[str], csv_filepath: Path) -> None:
        """
        Convert JSON data to CSV format.

        Parameters:
        - jsondata (List[str]): List of JSON strings.
        - fieldnames (List[str]): List of field names for the CSV.
        - csv_filepath (Path): Path to the CSV file.
        """
        data_dicts = [json.loads(data) for data in jsondata]
        with csv_filepath.open(mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for data in data_dicts:
                writer.writerow(data)

class likelihood_comp_Encoder(JSONEncoder):
    def default(self, o: Any) -> Dict[str, Any]:
        """
        Custom JSON encoder for likelihood_comparison objects.

        Parameters:
        - o (Any): Object to be encoded.

        Returns:
        - Dict[str, Any]: Dictionary representation of the object.
        """
        if isinstance(o, likelihood_comparison):
            return o.__dict__
        return super().default(o)  # delegates the serialization process to the standard JSONEncoder

