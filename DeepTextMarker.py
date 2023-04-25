import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from gensim import downloader
from gensim.models import Word2Vec
from gensim.parsing import preprocessing
import math
import string

class DeepTextMarker:
    """
    A class that implements the DeepTextMarker.
    
    ...
    
    Attributes
    ----------
    pretrained_word2vec : gensim model, optional
        The pretrained word2vec model. Must be from gensim. Defaults to the 'glove-twitter-200' model.
    sentence_encoder : tensorflow model, optional
        The pretrained sentence encoder. Should be a tensorflow object that produces sentence embeddings. 
        Defaults to the Unvisersal Sentence Encoder from ""https://tfhub.dev/google/universal-sentence-encoder/4".
        
    Methods
    -------
    watermark_single_sentence(given_sentence)
        When given a tokenized sentence (a list of tokens) returns the watermarked sentence as a list of tokens.
        
    watermark_multiple_sentences(sentences)
        When given a list of tokenized sentences (a list of token lists) returns the watermarked sentences as a list of tokenized sentences.
        
    display_result(given_sentence)
        When given a tokenized sentence (a list of tokens) prints all sentence proposals, their scores, and the sentence selected as the watermarked sentence.
    """
    
    def __init__(self, pretrained_word2vec_str = 'glove-twitter-200', sentence_encoder_str = "https://tfhub.dev/google/universal-sentence-encoder/4"):
        """
        Parameters
        ----------
        pretrained_word2vec : str, optional
            The string indicating the pretrained word2vec model to load. Defaults to the 'glove-twitter-200' model.
        sentence_encoder : str, optional
            The string indicating the tensorflow hub model to load.
            Default value is the string indicating the Unvisersal Sentence Encoder at "https://tfhub.dev/google/universal-sentence-encoder/4".
        """

        self.pretrained_word2vec = downloader.load(pretrained_word2vec_str)
        self.sentence_encoder = hub.load(sentence_encoder_str)
    
    def __remove_punct(self, given_str):
        return given_str.translate(str.maketrans('', '', string.punctuation))

    def __get_sentence_proposals(self, chosen_sentence):
        possible_syns = self.__get_possible_synonyms(chosen_sentence)
        sentence_proposals = []

        for i in range(len(possible_syns)):
            current_target = possible_syns[i]

            if current_target is not None:
                word = current_target[0]
                new_sent = chosen_sentence.copy()
                new_sent[i] = word
                sentence_proposals.append(new_sent)

        return sentence_proposals

    def __get_possible_synonyms(self, given_sentence, topn=1):
        possible_synonyms = []
        CUSTOM_FILTERS = [lambda x: x.lower(), preprocessing.strip_numeric, preprocessing.strip_punctuation, preprocessing.remove_stopwords, preprocessing.strip_multiple_whitespaces]
        preprocessed = [preprocessing.preprocess_string(word.lower(), CUSTOM_FILTERS) for word in given_sentence]

        for i in range(len(preprocessed)):
            word = preprocessed[i]

            if (len(word) != 1):
                possible_synonyms.append(None)
                continue

            try:
                similar_word = self.pretrained_word2vec.most_similar(word, topn=topn)[0]

                # Clear words that are already at the proposed location.
                if self.__remove_punct(given_sentence[i].lower()) == self.__remove_punct(similar_word[0].lower()):
                    possible_synonyms.append(None)
                    continue
                else:
                    possible_synonyms.append(similar_word)

            except KeyError:
                possible_synonyms.append(None)

        return possible_synonyms

    def __get_sentence_score(self, sent1, sent2):
        sts_encode1 = tf.nn.l2_normalize(self.sentence_encoder(sent1), axis=1)
        sts_encode2 = tf.nn.l2_normalize(self.sentence_encoder(sent2), axis=1)
        cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
        clip_cosine_similarities = tf.clip_by_value(cosine_similarities, -1.0, 1.0)
        scores = 1.0 - tf.acos(clip_cosine_similarities) / math.pi
        average = tf.math.reduce_mean(scores)
        return tf.get_static_value(average)

    def __get_score_list(self, given_sentence, proposals):
        scores = []

        for proposal_sent in proposals:
            scores.append(self.__get_sentence_score(given_sentence, proposal_sent))

        return scores
    
    def watermark_single_sentence(self, given_sentence):
        """
        When given a tokenized sentence (a list of tokens) returns the watermarked sentence as a list of tokens.
        
        Parameters
        ----------
        given_sentence : list of str
            The tokenized list of strings to watermark.
        
        Returns
        -------
        list of str 
            a list of strings that represent the watermarked sentence
        """

        proposals = self.__get_sentence_proposals(given_sentence)

        if len(proposals) == 0:
            return None

        scores = self.__get_score_list(given_sentence, proposals)

        return proposals[scores.index(max(scores))]
    
    def watermark_multiple_sentences(self, sentences):
        """
        When given a list of tokenized sentences (a list of token lists) returns the watermarked sentences as a list of tokenized sentences.
        
        Parameters
        ----------
        sentences : list of list of str
        
        Returns
        -------
        list of list of str
            a list of tokenized sentences that represent the watermarked sentences
        """
                 
        return [self.watermark_single_sentence(sentence) for sentence in sentences]
                 
    def display_result(self, given_sentence):
        """
        When given a tokenized sentence (a list of tokens) prints all sentence proposals, their scores, and the sentence selected as the watermarked sentence.
        
        Parameters
        ----------
        given_sentence : list of str
            The tokenized list of strings to watermark.
        """ 
                
        print("Original")
        print(given_sentence)
        print()

        replacements = [syn for syn in self.__get_possible_synonyms(given_sentence) if syn is not None]
        proposals = self.__get_sentence_proposals(given_sentence)
        scores = self.__get_score_list(given_sentence, proposals)

        print("Best Replacement")
        print("Index: " + str(scores.index(max(scores))))
        print(self.watermark_single_sentence(given_sentence))
        print()

        print("Proposals")
        for i in range(len(proposals)):
            proposal_sent = proposals[i]
            print(i)
            print(replacements[i])
            print(proposal_sent)
            print(self.__get_sentence_score(given_sentence, proposal_sent))
            print()