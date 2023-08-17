from transformers import BertModel, BertTokenizer
from scipy.spatial.distance import cosine
from scipy.spatial.distance import cdist

import torch
from icecream import ic
# from gensim.models import KeyedVectors
import numpy as np
from sentence_transformers import SentenceTransformer


class SynonymFinder_bert:
    def __init__(self, model_name='bert-base-uncased'):
        """
        Initialize with the specific transformer model.

        Args:
        model_name (str): The name of the transformer model.
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()  # Put the model in evaluation mode

    def find_synonyms(self, word, num_synonyms=10):
        """
        Find synonyms for a given word.

        Args:
        word (str): The word to find synonyms for.
        num_synonyms (int): The number of synonyms to return.

        Returns:
        list: List of synonyms.
        """
        word_vector = self._get_word_vector(word)
        closest = self._find_closest_words(word_vector, num_synonyms + 1)
        
        # remove the input word from the list if it exists
        if word in closest:
            closest.remove(word)

        return closest


    def _get_word_vector(self, word):
        """
        Get the vector representation of a word.

        Args:
        word (str): The word to vectorize.

        Returns:
        np.ndarray: The vector representation of the word.
        """
        inputs = self.tokenizer(word, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs.last_hidden_state[0, 0, :].detach().numpy()

    def _find_closest_words(self, word_vector, num_synonyms):
        """
        Find the closest words to a given word vector.

        Args:
        word_vector (np.ndarray): The vector representation of the word.
        num_synonyms (int): The number of synonyms to return.

        Returns:
        list: List of closest words.
        """
        closest_words = []
        for word, idx in self.tokenizer.get_vocab().items():
            word_emb = self.model.embeddings.word_embeddings.weight[idx].detach().numpy()
            distance = cosine(word_vector, word_emb)
            closest_words.append((distance, word))

        sorted_words = sorted(closest_words, key=lambda x: x[0])
        
        return [word for distance, word in sorted_words[:num_synonyms]]


class SynonymFinder_miniLM:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize with the specific transformer model.

        Args:
        model_name (str): The name of the transformer model.
        """
        self.model = SentenceTransformer(model_name)

    def find_synonyms(self, word, num_synonyms=5):
        """
        Find synonyms for a given word.

        Args:
        word (str): The word to find synonyms for.
        num_synonyms (int): The number of synonyms to return.

        Returns:
        list: List of synonyms.
        """
        word_vector = self.model.encode([word])[0]
        closest = self._find_closest_words(word_vector, num_synonyms + 1)

        # remove the input word from the list if it exists
        if word in closest:
            closest.remove(word)

        return closest

    def _find_closest_words(self, word_vector, num_synonyms):
        """
        Find the closest words to a given word vector.

        Args:
        word_vector (np.ndarray): The vector representation of the word.
        num_synonyms (int): The number of synonyms to return.

        Returns:
        list: List of closest words.
        """
        embeddings = []
        words = []

        # encode words in batches to speed up the process
        batch_size = 1000
        for idx in range(0, len(self.model.tokenizer.get_vocab()), batch_size):
            word_batch = list(self.model.tokenizer.get_vocab().keys())[idx:idx+batch_size]
            word_embeddings = self.model.encode(word_batch)
            embeddings.extend(word_embeddings)
            words.extend(word_batch)

        embeddings = np.array(embeddings)
        distances = cdist([word_vector], embeddings, "cosine")[0]

        closest_idxs = np.argsort(distances)[:num_synonyms]  # get indices of closest words

        return [words[idx] for idx in closest_idxs]

if __name__ == '__main__':
    synonym_finder = SynonymFinder_miniLM()
    words = ['intelligence', 'love', 'hate']
    synonyms = {word: synonym_finder.find_synonyms(word) for word in words}

    ic(synonyms)