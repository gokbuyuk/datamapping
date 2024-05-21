# 
README



# Semantic Data Mapping

This repository contains code for computing and determining matches between source and target variables based on semantic similarity metrics. The code is written in Python 3.11.3. Below are descriptions of the major code files in this repository:

### 1. `get_matching.py`

This code holds the primary function of this repository, `get_matching`. This function establishes the matching between the source and target variables leveraging the similarity in their names, categories, descriptions, and data types.

 **Usage** :
To utilize this function, one must specify the following parameters:

* `fileprefix`: Prefix for all generated files.
* `source_data_desc_filepath`: Path to the source data description file (has variable names, categories, descriptions, and data types).
* `input_variable_format`: A list of variable names in order of Variable name, Variable category, Variable description, Variable data type.
* `target_data_desc_path`: Path to the target data description file.
* `output_data_path`: Path where the output will be saved.

 **Outputs** :
The *processed* folder of the output directory will contain:

1. CSV with matching between source and target using the source as the reference.
2. CSV with matching using target as the reference.
3. JSON file with matching dictionary (target as key).

For those with available ground truth, the reports below can be found in the *reports folder of the given output directory*:

4. CSV matching report using source as key.
5. CSV matching report using target as key.


### 2. `list_similarity.py`

This file encapsulates a suite of functions designed to compute the semantic similarity between lists. The underlying methodology employs sentence transformers for word vectorization and leverages cosine similarity for semantic matching.

 **Functions** :

* `get_semantic_similarity`: Calculates semantic similarity between two phrases via sentence transformers.
* `compute_syntactic_similarity`: Evaluates syntactic similarity between two variable descriptions using SequenceMatcher.
* `create_similarity_matrix`: Forms a similarity matrix between two phrase lists.
* `create_similarity_table`: Establishes a similarity table between two phrase lists.
* `find_top_similar_items`: Given two lists, this function outputs a dictionary. Keys derive from the first list, and values represent the top matches from the second list, ensuring no repetitions.
* `table_to_string`: A utility to transform a table (list of lists) into a string format.

### Others:

The other python files contain other helper functions to the main function.

### How do I get set up?

* Install the libraries listed in requirements.txt

### Contribution guidelines

Contributions are welcome! You can help with:

* Writing tests
* Improving similarity measure
* Code review
