"""This file contains functions to convert a matching dictionary to a dataframe and evaluating the results if the grund truth is available
"""

import pandas as pd
import unittest 
import os
import json
from icecream import ic

def predictions_to_dataframe(data_list):
    """
    Convert a list of dictionaries representing predictions to a DataFrame.

    Args:
        data_list (List[dict]): A list of dictionaries, where each dictionary contains the prediction information.

    Returns:
        pd.DataFrame: A DataFrame with columns 'state' and 'fda', containing the mappings from the input list.
    """
    # Initialize lists to store data for each column
    state_data = []
    fda_data = []

    # Iterate through the list of dictionaries
    for item in data_list:
        category_name = item['category_name']
        mappings = item['mappings']
        mappings = [mapping for mapping in mappings if len(mapping) > 0]
        for mapping in mappings:
            # ic(mapping, len(mapping))
            fda_element = mapping['fda_element']
            state_ele_recommended = mapping['state_ele_recommended'][0]

            # Append values to the corresponding lists
            state_data.append(state_ele_recommended)
            fda_data.append(fda_element)

    # Create the dataframe
    df = pd.DataFrame({'state': state_data, 'fda': fda_data})
    return df

class TestPredictionsToDataFrame(unittest.TestCase):

    def test_predictions_to_dataframe(self):
        data_list = [
            {
                'category_name': 'Social',
                'mappings': [
                    {
                        'fda_desc': 'Poverty rate',
                        'fda_dtype': 'Numeric',
                        'fda_element': 'Pov',
                        'state_ele_desc': ['Poverty rate', 'Area Deprivation Index'],
                        'state_ele_recommended': ['Poverty', 'AreaDepIdx']
                    },
                    {
                        'fda_desc': 'Area Deprivation Index',
                        'fda_dtype': 'Numeric',
                        'fda_element': 'ADI',
                        'state_ele_desc': ['Area Deprivation Index', 'Poverty rate'],
                        'state_ele_recommended': ['AreaDepIdx', 'Poverty']
                    }
                ]
            }
        ]
        
        expected_df = pd.DataFrame({
            'state': ['Poverty', 'AreaDepIdx'],
            'fda': ['Pov', 'ADI']
        })
        
        result_df = predictions_to_dataframe(data_list)
        
        self.assertTrue(expected_df.equals(result_df))


def evaluate_example():
    GROUND_TRUTH_PATH = os.path.join('data', 'raw', 'state0_ground_truth.csv')  # Set this to None if there is no groundtruth
    PREDICTION_PATH = os.path.join('data', 'processed', 'teststate_v3__matching_dict.json')
    state0_ground_truth = pd.read_csv(GROUND_TRUTH_PATH, header=0)
    state0_ground_truth.columns = ['state', 'fda']
    ic(state0_ground_truth)
    with open(PREDICTION_PATH, 'r') as f:
        predictions = json.load(f)
    # ic(predictions)
    state0_predictions = predictions_to_dataframe(predictions)
    ic(state0_predictions)
    state_to_fda_report = pd.merge(state0_ground_truth, state0_predictions, 
                                   how='outer', on='fda', 
                                   suffixes=('_ground_truth', '_prediction'))
    state_to_fda_report.dropna(subset=['state_prediction'], inplace=True)
    state_to_fda_report['correct_match'] = state_to_fda_report['state_ground_truth'] == state_to_fda_report['state_prediction']
    cols_order = ['fda', 'state_prediction', 'state_ground_truth', 'correct_match']
    state_to_fda_report = state_to_fda_report[cols_order]
    ic(state_to_fda_report.sort_values(by=['correct_match']))
    ic(state_to_fda_report['correct_match'].mean().round(3))
    state_to_fda_report.to_csv(os.path.join('reports', 'teststate_v3__matching_dict_report.csv'), index=False)
if __name__ == '__main__':
    evaluate_example()
    # unittest.main()
    
