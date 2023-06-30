import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from gensim import downloader
from gensim.models import Word2Vec
from gensim.parsing import preprocessing
from nltk.tokenize import sent_tokenize
from nltk.metrics.distance import edit_distance
from nltk.tokenize.treebank import TreebankWordDetokenizer, TreebankWordTokenizer
import math
import string

class DeepTextMarker:
    """
    A class that implements the DeepTextMarker.
    
    ...
    
    Attributes
    ----------
    pretrained_word2vec : gensim model
        The pretrained word2vec model. Must be from gensim. Defaults to the 'glove-twitter-200' model.
    sentence_encoder : tensorflow model
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

    def __percentage_word_similarity(self, word1, word2):
        levenshtein_distance = edit_distance(word1, word2)
        return 1 - levenshtein_distance / max(len(word1), len(word1))
    
    def __get_sentence_proposals(self, chosen_sentence, topn=1):
        possible_syns = self.__get_possible_synonyms(chosen_sentence, topn)
        
        sentence_proposals = []

        for possible_syn in possible_syns:
            if possible_syn is not None:
                word, index = possible_syn
                
                new_sent = chosen_sentence.copy()
                
                original_word = new_sent[index]
                new_sent[index] = word
                
                sentence_proposals.append((new_sent, word, original_word))

        return sentence_proposals

    def __get_possible_synonyms(self, given_sentence, topn=1):
        possible_synonyms = []
        CUSTOM_FILTERS = [lambda x: x.lower(), preprocessing.strip_numeric, preprocessing.strip_punctuation, preprocessing.remove_stopwords, preprocessing.strip_multiple_whitespaces]
        preprocessed = [preprocessing.preprocess_string(word.lower(), CUSTOM_FILTERS) for word in given_sentence]

        for i in range(len(preprocessed)):
            word = preprocessed[i]

            if (len(word) != 1):
                continue

            try:
                similar_words = self.pretrained_word2vec.most_similar(word, topn=topn)
                original_token = self.__remove_punct(given_sentence[i].lower())
                
                for similar_word in similar_words:
                    # similar_word is currently a word and a score, get just the word.
                    similar_word = similar_word[0]
                    
                    # Clear words that are already at the proposed location. 
                    if original_token == self.__remove_punct(similar_word.lower()):
                        continue
                    else:
                        # Append the synonym and its replacement location here.
                        possible_synonyms.append((similar_word, i))

            except KeyError:
                continue

        return possible_synonyms

    def __get_sentence_score(self, sent1, proposal_sent):
        sent2, proposal_word, original_word = proposal_sent
        
        sts_encode1 = tf.nn.l2_normalize(self.sentence_encoder(sent1), axis=1)
        sts_encode2 = tf.nn.l2_normalize(self.sentence_encoder(sent2), axis=1)
        cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
        clip_cosine_similarities = tf.clip_by_value(cosine_similarities, -1.0, 1.0)
        scores = 1.0 - tf.acos(clip_cosine_similarities) / math.pi
        average = tf.math.reduce_mean(scores)
        average = tf.get_static_value(average)
        
        # Maximize the word difference and sentence similarity to reduce number of trivial substitutions.
        # word_similarity = self.__percentage_word_similarity(proposal_word, original_word)
        #score = average - word_similarity
        
        score = average
        
        return score

    def __get_score_list(self, given_sentence, proposals):
        scores = []

        for proposal_sent in proposals:
            scores.append(self.__get_sentence_score(given_sentence, proposal_sent))

        return scores
    
    def watermark_single_sentence(self, given_sentence, topn=1):
        """
        When given a tokenized sentence (a list of tokens) returns the watermarked sentence as a list of tokens.
        
        Parameters
        ----------
        given_sentence : list of str
            The tokenized list of strings to watermark.
        
        topn : int, optional 
            The number of synonyms to consider per word.
        
        Returns
        -------
        list of str 
            a list of strings that represent the watermarked sentence
        """

        proposals = self.__get_sentence_proposals(given_sentence, topn)
        
        if len(proposals) == 0:
            return None

        scores = self.__get_score_list(given_sentence, proposals)

        watermarked, _, _ = proposals[scores.index(max(scores))]
        
        return watermarked
    
    def watermark_multiple_sentences(self, sentences, topn=1):
        """
        When given a list of tokenized sentences (a list of token lists) returns the watermarked sentences as a list of tokenized sentences.
        
        Parameters
        ----------
        sentences : list of list of str
            The sentences to watermark.
        
        topn : int, optional 
            The number of synonyms to consider per word.
        
        Returns
        -------
        list of list of str
            a list of tokenized sentences that represent the watermarked sentences
        """
                 
        return [self.watermark_single_sentence(sentence, topn) for sentence in sentences]
                
    def tokenize_and_watermark(self, text, topn=1):
        """
        When given a string that represents text, this function tokenizes and watermarkes the text.
        
        Parameters
        ----------
        text : string
            The text to watermark.
        
        topn : int, optional 
            The number of synonyms to consider per word.
        
        Returns
        -------
        list of string
            a list of watermarked sentences
        """
        
        # Tokenize the input string.
        tokenized_sents = [sent.lower() for sent in sent_tokenize(text)]
        word_tokenizer = TreebankWordTokenizer()
        full_tokenized = [word_tokenizer.tokenize(sentence) for sentence in tokenized_sents]
        
        # Predict on the tokenized data.
        watermarked_text = self.watermark_multiple_sentences(full_tokenized, topn)
        
        # Detokenize and return.
        detokenizer = TreebankWordDetokenizer()
        return [detokenizer.detokenize(sentence) for sentence in watermarked_text] 
    
    def prepredict_watermarking(self, detector, text, topn=1, print_ratio=False):
        """
        This watermarking method prepredicts on sentences, and only embeds watermarks that are successfully detected.
        If the detection does not succeed on a majority of the sentences, the watermarking fails.
        
        Parameters
        ----------
        detector : Tensorflow Model
            The DeepTextMarkDetector to use
            
        text : string
            the string to watermark
        
        Returns
        -------
        list of string or none
            returns the list of string if the watermarking is successful, otherwise returns none
        """
        
        # Tokenize the input string.
        tokenized_sents = [sent.lower() for sent in sent_tokenize(text)]
        num_sents = len(tokenized_sents)
        word_tokenizer = TreebankWordTokenizer()
        full_tokenized = [word_tokenizer.tokenize(sentence) for sentence in tokenized_sents]
        detokenizer = TreebankWordDetokenizer()
        result = []
        num_detected = 0
        
        for sent in full_tokenized:
            detokenized_watermarked = detokenizer.detokenize(self.watermark_single_sentence(sent, topn))
            
            # Get the single value that represents the detection result.
            result_val = detector.predict([detokenized_watermarked])[0][0]
            
            # Case where watermark detection is unsuccessful.
            if result_val < .5: 
                # Append the original sentence.
                result.append(detokenizer.detokenize(sent))
            # Case where watermark detection is successful.
            else:
                # Append the watermarked sentence
                result.append(detokenized_watermarked)
                num_detected += 1
           
        watermarked_ratio = num_detected / num_sents
        
        if print_ratio:
            print("Watermarked ratio: " + str(watermarked_ratio))
            
        if watermarked_ratio < .51:
            return None
        else:
            return result
    
    def display_result(self, given_sentence):
        """
        When given a tokenized sentence (a list of tokens) prints all sentence proposals, their scores, and the sentence selected as the watermarked sentence.
        
        Parameters
        ----------
        given_sentence : list of str
            The tokenized list of strings to watermark.
        """ 
        
        detokenizer = TreebankWordDetokenizer()
        
        print("Original")
        print(detokenizer.detokenize(given_sentence))
        print()

        replacements = [syn for syn in self.__get_possible_synonyms(given_sentence, 1) if syn is not None]
        proposals = self.__get_sentence_proposals(given_sentence)
        scores = self.__get_score_list(given_sentence, proposals)
        
        
        print("Best Replacement")
        print("Proposal Number: " + str(scores.index(max(scores))))
        print(detokenizer.detokenize(self.watermark_single_sentence(given_sentence)))
        print()

        print("Proposals")
        for i in range(len(proposals)):
            proposal_sent, _, _ = proposals[i]
            print("Proposal Number: " + str(i))
            
            word, replacement_location = replacements[i]
            
            print("Replacement Word: " + word)
            print("Replacement Location: index " + str(replacement_location)) 
            print("Score: " + str(self.__get_sentence_score(given_sentence, proposals[i])))
            print(detokenizer.detokenize(proposal_sent))
            print()