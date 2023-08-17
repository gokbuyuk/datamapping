"""This file contains functions to compute semantic similarity between two lists, create similarity matrix, \
    and other functions to create similarity table and find top similar items between two lists.\
    The words are vectorized using sentence transformers.\
    The semantic similarity matrix is computed using sentence transformers and cosine similarity.\
    The functions are listed below:
        1. get_semantic_similarity: Compute the semantic similarity between two phrases using sentence transformers.
        2. compute_syntactic_similarity: Compute the syntactic similarity between two variable descriptions using SequenceMatcher.
        3. create_similarity_matrix: Create a similarity matrix between two lists of phrases. 
        4. create_similarity_table: Create a similarity table between two lists of phrases.
        5. find_top_similar_items: This function takes two lists and returns a dictionary with key, value pairs where keys are the items
            from the first list, and values are the top most similar items from the second list, without duplicates.
        6. table_to_string: Convert table (list of lists) to a string
        
    
"""

import pandas as pd
from icecream import ic
import os
import openai
from tablesimilarity import *
# for semantic similarity:
from sentence_transformers import SentenceTransformer, util
from scipy.spatial.distance import cosine
from difflib import SequenceMatcher

import numpy as np
import unittest
from typing import List, Dict, Tuple

openai.organization = "org-2hsw3oztN5UIEExVcOPdxIZq"
openai.api_key_path = "API_key/openAI"

openai.api_key = os.getenv("OPENAI_API_KEY")

model='paraphrase-MiniLM-L6-v2'

# Convert table (list of lists) to a string
def table_to_string(table):
    return "\t".join("|".join(map(str, row)) for row in table)

class TestTableToString(unittest.TestCase):
    def test_table_to_string(self):
        table = [['A', 1, 2], ['B', 4, 5], ['C', 7, 8]]
        result = table_to_string(table)
        # ic(result)
        expected_result = 'A|1|2\tB|4|5\tC|7|8'
        self.assertEqual(result, expected_result)

###### get semantic similarity
def get_semantic_similarity(text1, text2, model='paraphrase-MiniLM-L6-v2'):
    """
    Compute the semantic similarity between two phrases using sentence transformers.

    Args:
        text1 (str): First phrase.
        text2 (str): Second phrase.
        model (str, optional): Name of the sentence transformer model to use. Defaults to 'paraphrase-MiniLM-L6-v2'.

    Returns:
        float: Semantic similarity score between the two phrases.
    """
    
    model = SentenceTransformer(model)
    # Generate embeddings for both phrases
    embeddings = model.encode([text1, text2])
    # ic(type(embeddings), embeddings)
    # Compute the cosine similarity between the embeddings
    similarity = 1 - cosine(embeddings[0], embeddings[1])

    return similarity

def compute_syntactic_similarity(description1, description2):
    """
    Compute the syntactic similarity between two variable descriptions using SequenceMatcher.

    Args:
        description1 (str): The first variable description.
        description2 (str): The second variable description.

    Returns:
        float: The syntactic similarity score between the variable descriptions.
    """
    return SequenceMatcher(None, description1, description2).ratio()


def get_table_semantic_similarity(table1, table2,  model='paraphrase-MiniLM-L6-v2'):
    """get semantic similarity between two tables when tables are given as a list of column values or row values
    e.g. table1 = [['col1', a', 'b', 'c'], ['col2', 'd', 'e', 'f']]

    Args:
        table1 (list): first table in list of lists format
        table2 (list): second table in list of lists format
        model (str, optional): embedding model. Defaults to 'paraphrase-MiniLM-L6-v2'.
    """    
    # convert table to string
    table1_str = table_to_string(table1)
    table2_str = table_to_string(table2)
    
    # Get the semantic similarity between the two tables
    similarity = get_semantic_similarity(table1_str, table2_str, model=model)
    assert 0 <= similarity <= 1, "Similarity score should be between 0 and 1."
    return similarity


def create_similarity_matrix(list1, list2, 
                             similarity = 'semantic',
                             weights = None,
                             model = 'paraphrase-MiniLM-L6-v2'):
    # Initialize an empty similarity matrix
    # similarity_matrix = np.zeros((len(list1), len(list2)))

    # # Compute the semantic similarity for each pair of phrases
    # results = Parallel(n_jobs=-1)(delayed(compute_similarity)(i, phrase1, list2, similarity, weights, model) for i, phrase1 in enumerate(list1))

    # for i, similarities in results:
    #     similarity_matrix[i, :] = similarities
    if isinstance(model,str):
        model = SentenceTransformer(model)
    embeddings1 = model.encode(list1)
    embeddings2 = model.encode(list2)

    #Compute cosine-similarities for each sentence with each other sentence

    cosine_scores = util.cos_sim(embeddings1, embeddings2).numpy()
    cosine_scores = np.clip(cosine_scores, a_max=1.0, a_min=-1.0)
    assert len(list1), len(list2) == cosine_scores.shape
    # similarity_matrix = np.zeros((len(list1), len(list2)))
    # print("Cosine scores:", cosine_scores)
    # print("similarity matrix:", similarity_matrix)
    return cosine_scores


class TestCreateSimilarityMatrix(unittest.TestCase):
    def test_create_similarity_matrix(self):
        list1 = ["sample_id", "user_name", "registration_date"]
        list2 = ["id_of_sample", "name_of_user", "date_of_registration"]
        # ic(list1, list2)
        self_similarity_matrix = create_similarity_matrix(list1, list1)
        # print("Created self-similarithy matrix:")
        # print(self_similarity_matrix)
        
        self.assertEqual(self_similarity_matrix.shape, (3, 3))
        self.assertTrue(np.all(0 <= self_similarity_matrix))
        self.assertTrue(np.all(self_similarity_matrix <= 1.001))
        self.assertTrue(np.isclose(self_similarity_matrix[0, 0], 1.0))
        self.assertTrue(np.isclose(self_similarity_matrix[1, 1], 1.0))
        self.assertTrue(np.isclose(self_similarity_matrix[2, 2], 1.0))
        
        similarity_matrix = create_similarity_matrix(list1, list2)
        # print("Created similarithy matrix:")
        # print(similarity_matrix)
        
        self.assertEqual(similarity_matrix.shape, (3, 3))
        self.assertTrue(np.all(0 <= similarity_matrix) & np.all(similarity_matrix <= 1))
        

def create_similarity_table(list1, list2, similarity_matrix=None, model = 'paraphrase-MiniLM-L6-v2', one_to_one=True):
    """Create a similarity table between two lists of phrases.

    Args:
        list1 (lst): first list of phrases
        list2 (lst): second list of phrases
        similarity_matrix (array, optional): similarity matrix. Defaults to None.
        model (str, optional): Name of the sentence transformer model to use. Defaults to 'paraphrase-MiniLM-L6-v2'.
        one_to_one (bool, optional): whether to allow one-to-many relationship. \
            Defaults to True, allowing one-to-one matching: not allowing multiple items in list2 to match with the same item in list1.
    Returns:
        DataFrame: 
    """    
    return_dict = dict()
    if similarity_matrix is None:
        similarity_matrix = create_similarity_matrix(list1, list2)
        return_dict['similarity_matrix'] = similarity_matrix
    
    similarity_df = pd.DataFrame(similarity_matrix, index=list1, columns=list2)
    sorted_similarity_df = similarity_df.apply(lambda row: row.sort_values(ascending=False).index, axis=1)
    
    result = pd.DataFrame({'df1': sorted_similarity_df.index, 'df2': sorted_similarity_df.apply(list)})
    return_dict['matching_table'] = result.reset_index(drop=True)
    
    return return_dict
       
class TestCreateSimilarityTable(unittest.TestCase):

    def test_create_similarity_table(self):
        list1 = ['item1', 'item2', 'item3']
        list2 = ['title1', 'title2', 'title3']
        
        # Example similarity matrix, you should replace this with your actual similarity matrix
        similarity_matrix = np.array([[0.9, 0.2, 0.1],
                                      [0.1, 0.8, 0.5],
                                      [0.3, 0.6, 0.7]])
        
        result = create_similarity_table(list1, list2, similarity_matrix)
        result = result['matching_table'].reset_index(drop=True)
        
        expected_result = pd.DataFrame({'df1': ['item1', 'item2', 'item3'],
                                        'df2': [['title1', 'title2', 'title3'],
                                                  ['title2', 'title3', 'title1'],
                                                  ['title3', 'title2', 'title1']]})
        # ic(result)
        # ic(expected_result)     
        pd.testing.assert_frame_equal(result, expected_result, check_like=False)

def find_top_similar_items(list1: List[str], list2: List[str]) -> Dict[str, str]:
    """
    This function takes two lists and returns a dictionary with key, value pairs where keys are the items
    from the first list, and values are the top most similar items from the second list, without duplicates.

    Args:
        list1 (List[str]): The first list of items.
        list2 (List[str]): The second list of items.

    Returns:
        Dict[str, str]: A dictionary where keys are items from the first list, and values are the top most similar
                        items from the second list, without duplicates.
    """
    # assert len(list1) <= len(list2), "The length of list1 should be less than or equal to the length of list2."

    # similarity_matrix = create_similarity_matrix(list1, list2)
    # sorted_indices = np.argsort(similarity_matrix, axis=1)[:, ::-1]

    # assigned = set()
    # result = {}
    # for index, row in enumerate(sorted_indices):
    #     for col in row:
    #         if list2[col] not in assigned:
    #             result[list1[index]] = list2[col]
    #             assigned.add(list2[col])
    #             break
    # result_df = pd.DataFrame({'df1': list(result.keys()), 'df2': list(result.values())}).reset_index(drop=True)

    assert len(list1) <= len(list2), "The length of list1 should be less than or equal to the length of list2."

    similarity_matrix = create_similarity_matrix(list1, list2)
    row_max_indices = np.argmax(similarity_matrix, axis=1).tolist()
    # ic(similarity_matrix, row_max_indices)
    result = {}
    for i, max_index in enumerate(row_max_indices):
        result[list1[i]] = list2[max_index]

    result_df = pd.DataFrame({'df1': list(result.keys()), 'df2': list(result.values())}).reset_index(drop=True)
    # ic(result_df)
    return result_df

class TestFindTopSimilarItems(unittest.TestCase):
    def test_find_top_similar_items(self):
        list1 = ["sample_id", "animal", "weight", "age"]
        list2 = ["id_of_sample", "species", "mass", "oldness"]
        expected_dict = {
            "sample_id": "id_of_sample",
            "animal": "species",
            "weight": "mass",
            "age": "oldness"
        }
        expected_output = pd.DataFrame({'df1': list(expected_dict.keys()), 'df2': list(expected_dict.values())})
        # ic(expected_output)
        result = find_top_similar_items(list1, list2)
        ic(result)
        pd.testing.assert_frame_equal(result, expected_output, check_like=False)


        list1 = ["apple", "car", "tree"]
        list2 = ["automobile", "plant", "fruit"]

        expected_dict = {
            "apple": "fruit",
            "car": "automobile",
            "tree": "plant"
        }
        expected_output = pd.DataFrame({'df1': list(expected_dict.keys()), 'df2': list(expected_dict.values())})

        result = find_top_similar_items(list1, list2)
        pd.testing.assert_frame_equal(result, expected_output, check_like=False)


        # Test case with same items in both lists
        list1 = ["apple", "car", "tree"]
        list2 = ["apple", "car", "tree"]

        expected_dict = {
            "apple": "apple",
            "car": "car",
            "tree": "tree"
        }
        expected_output = pd.DataFrame({'df1': list(expected_dict.keys()), 'df2': list(expected_dict.values())})

        result = find_top_similar_items(list1, list2)
        pd.testing.assert_frame_equal(result, expected_output, check_like=False)

if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)

