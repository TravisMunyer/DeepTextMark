import tensorflow as tf
from nltk.tokenize import sent_tokenize

class DeepTextMarkDetector:
    """
    A class that detects strings as DeepTextMark watermarked or not.
    
    ...
    Attributes
    ----------
    detector : tensorflow model
        The trained classifier for DeepTextMarkDetector.
        
    Methods
    -------
    predict_sentences(text)
        When given text as a string, this method sentence tokenizes the input, and classifies the entirety of the text as watermarked or not.
    
    detect_percentage_watermarked(text):
        Predicts the percentage of text that is watermarked.
    """
    
    def __init__(self, classifier_location):
        """
        Parameters
        ----------
        classifier_location : string
            the file path of the trained DeepTextMarkDetector tensorflow model to load.
        """ 
        
        self.detector = tf.keras.models.load_model(classifier_location)
        
    def predict_sentences(self, text):
        """
        text : string
            The text to perform prediction on.
        """
        
        tokenized_text = [sent.lower() for sent in sent_tokenize(text)]
        
        # Threshold and prepare the predictions.
        predictions = self.detector.predict(tokenized_text)
        predictions[predictions >= .5] = 1
        predictions[predictions < .5] = 0
        predictions = list(map(int, predictions))

        # Get the count of predictions that are watermarked (equal to 1).
        watermarked_count = predictions.count(1)

        # If more than half of the predictions are watermarked, return watermarked.
        if(watermarked_count > len(predictions)/2):
            return "watermarked"
        else:
            return "unmarked"
        
    def detect_percentage_watermarked(self, text):
        """
        text : string
            The text to perform prediction on.
        """
        
        tokenized_text = [sent.lower() for sent in sent_tokenize(text)]
        
        # Threshold and prepare the predictions.
        predictions = self.detector.predict(tokenized_text)
        predictions[predictions >= .5] = 1
        predictions[predictions < .5] = 0
        predictions = list(map(int, predictions))
        
        # Get the count of predictions that are watermarked (equal to 1).
        watermarked_count = predictions.count(1)
        
        return watermarked_count / len(tokenized_text)