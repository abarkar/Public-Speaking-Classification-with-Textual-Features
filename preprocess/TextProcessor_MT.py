"""
This script extracts textual features based on:
@article{Vajjala2016AutomatedAO,
  title={Automated Assessment of Non-Native Learner Essays: Investigating the Role of Linguistic Features},
  author={Sowmya Vajjala},
  journal={International Journal of Artificial Intelligence in Education},
  year={2016},
  volume={28},
  pages={79 - 105},
  url={https://api.semanticscholar.org/CorpusID:3666482}
}



@author: Alisa Barkar, alisa.george.barkar@gmail.com
"""
# NLP
from flair.data import Sentence
from flair.models import SequenceTagger
import spacy
import gensim.models as g
from nltk.corpus import wordnet
import nltk
# System
import argparse
import sys
import os
import codecs
from tqdm import tqdm
# Data & Mathmatics
import numpy as np
import pandas as pd
import math
# Import the config reader
from config_reader import read_config
# Global
global featureCategories
global dataset
global language


#---------------------------------------------------------------------
# Manipulation with french and english POSTagging
#---------------------------------------------------------------------
def nbTokens(doc):
    return len(doc)

def nbType(doc):
    types = tokenTypes(doc)
    return len(types)

def tokenTypes(doc):
    types = dict()
    for token in doc:
        x = types.setdefault(token.pos_)
        if x == None:
            types[token.pos_] = 1
        else:
            types[token.pos_] += 1
    return types


# This function takes doc = nlp(transcript) with fr_core_news_sm
# And gives a list of lists where each one is (word, lemma, POS, general tag)
#TODO: lowercase !!!! for lemmas at least!!
class frencToken():
    """
    Class of french tokens. 

    Attributes:
    self.text (str): word.  
    self.lemma_ (str): lemma of the word
    self.pos_ (str): pos of the word
    self.tag_ (str): tag of the word
    """
    def __init__(self, text = None, lemma = None, pos = None, tag = None):
        self.text = text
        self.lemma_ = lemma
        self.pos_ = pos
        self.tag_ = tag

class frenchDoc():
    """
    Class of french document postagging. 

    Attributes:
    self.text (str): text of the transcript.  
    self.docFull (list of lists): list of lists [spacyToken.text, spacyToken.lemma_, spacyToken.pos_, tag]
    self.docPerSent (list of lists of lists): list of lists corresponding the sentences of the text where each is the sequence of lists [spacyToken.text, spacyToken.lemma_, spacyToken.pos_, tag]
    """
    def __init__(self, listOfFrenchSentences):
        self.docFull = []
        self.text = ''
        for sent in listOfFrenchSentences:
            self.docFull.extend(sent)
            for word in sent:
                self.text += word.text + ' '
        
        self.docPerSent = listOfFrenchSentences


def sentenceSeparation(transcript):
    """
    Function is used for separation on sentences in the case if there is no punctuation. 
    With whisper --- no need to use, it is commented. 

    Parameters:
    transcript (srt): full text of the .txt file. 

    Returns:
    sentences (list of strings): list of sentences of the text.
    """
    #TODO: because of automatic annotation there is no sentences
    # Time: 180 sec
    # average length of sentence in french (words): 10-15
    transcript = transcript.split(" ")
    nbOfWords = len(transcript)
    end = round(nbOfWords/10)
    sentences = []

    for i in range(0, end):
        if (i != end ):
            sentByWord = transcript[i*10: (i+ 1)*10]
        else: 
            sentByWord = transcript[i:]
        
        sentences.append(" ".join(sentByWord))
    return sentences


def fullTaggingForFrench(transcript):
    """
    Function that gives frenchDoc class object for the given text. 

    Parameters:
    transcript (srt): full text of the .txt file. 

    Returns:
    doc (frenchDoc class object): object with three main attributes .docFull .docPerSent and .text
    """

    # Load the model
    model = SequenceTagger.load("qanastek/pos-french")
    nlp = spacy.load('fr_core_news_sm')
   
    # List of sentences
    taggedDocBySent = []
    # Separation on sentences if there is NO punctuation in transcript:
    # sentences = sentenceSeparation(transcript)

    # Separation on sentences if there is punctuation in transcript:
    # sentences = transcript.split('.')
    sentences = [s.strip() for s in transcript.split('.') if s.strip()]
    #print("____________________________________________\n")
    #print(sentences)
    #print("____________________________________________\n")
    listOfSentences = []

    for sentence in sentences:
        # Create sentence object out of doc
        string = Sentence(sentence)
        # Predict tags
        prediction = model.predict(string)
        # Split tagged object to get list of pairs word-tag
        #print(string, "\n")
        taggedText = string.to_tagged_string().split("→")[1].split(", ")
        # Creates a list of pairs (token_text, token_tag)
        taggedText = [words.replace(' ', '').replace('\"', '').replace('\\', '').replace('[', '').replace(']', '').split("/")[1] for words in taggedText]
        # extract POS via spacy 
        spacyPOS = nlp(sentence)
        # Create a list of tokens in the sentence
        listOfTokens = [ frencToken(spacyToken.text, spacyToken.lemma_, spacyToken.pos_, tag) for tag, spacyToken in zip(taggedText, spacyPOS)]
        listOfSentences.append(listOfTokens)

    doc = frenchDoc(listOfSentences)
    return doc

#---------------------------------------------------------------------
# Preprocess raw transcript and split it into sentences
#---------------------------------------------------------------------


def text2sentence(transcript_dir, id):
    """
    Function that downloads text from the transcript and launches the creation
    of the new frenchDoc class object. 

    Parameters:
    ID (srt): ID of .txt file in the directory with transcripts. 

    Returns:
    doc (frenchDoc class object): object with three main attributes .docFull .docPerSent and .text
    """
    file = os.join(transcript_dir, id+".txt")
    with codecs.open(file, "r", "utf-8") as f:
        lines = f.readlines()
        text = ''
        for l in lines:
            tmp = l.strip()
            if len(tmp) != 0:
                text += tmp + ' '
    nlp = spacy.load('fr_core_news_sm')
    #nlp.pipe_labels['tagger']

    if (language == 'french'):
        doc = fullTaggingForFrench(text)
    else: 
        doc = nlp(text)

    print(text)

    # #print([token.tag_ for token in doc ])
    # test_docs = [sent.text.split() for sent in doc.sents]
    return doc


#---------------------------------------------------------------------
# Word Level Features of Lexical diversity
#---------------------------------------------------------------------
# Those features are taken from article
# paper: Automated assessment of non-native learner essays: Investigating the role of linguistic features
# authors: Sowmya Vajjala, Iowa State University, USA sowmya@iastate.edu

def measureOfTextualLexicalDiversity(doc):
    """
    Function for MTLD calculation. 
    
    Parameters:
    doc (frenchDoc class oblect): contains .docFull wich is the list of lists [word, word.lemma, word.pos, word.tag]

    Returns:
    (float): MTLD value.
    """
    lemmasList = [token.lemma_ for token in doc]

    types = dict()
    factors = 0
    ttrThreshold = 0.72
    startIndex = 0
    ttr = 1
    i = 0

    for token in lemmasList:
        types.setdefault(token)
        ttr = len(types)/ (i+1-startIndex)
        if (ttr < 0.72):
            startIndex = i+1
            types.clear()
            factors += 1
        elif (ttr > ttrThreshold and i == len(lemmasList)-1):
            factors += (1 - ttr) / (1 - 0.72)
        i += 1
    mtld1 = len(lemmasList) / factors
    factors = 0
    startIndex = len(lemmasList) - 1
    ttr = 1
    i = len(lemmasList) - 1
    types.clear()


    for token in list(reversed(lemmasList)):
        types.setdefault(token)
        ttr = len(types) / (startIndex - i+1)
        if (ttr < 0.72):
            startIndex = i-1
            types.clear()
            factors += 1
        elif (ttr > ttrThreshold and i == len(lemmasList)-1):
            factors += (1 - ttr) / (1 - 0.72)
        i -= 1

    mtld2 = len(lemmasList) / factors
    res = (mtld1 + mtld2) / 2
    if not math.isfinite(res):
        return 0.0
    else:
        return res


def lexicalDiversity(doc):
    """
    Function for lexical diversity feature calculation. 
    
    Parameters:
    doc (frenchDoc class oblect): contains .docFull wich is the list of lists [word, word.lemma, word.pos, word.tag]

    Returns:
    (list): list of features.
    """
    nb_types = nbType(doc)
    nb_tokens = nbTokens(doc)
    TTR = nb_types/nb_tokens
    CorrectedTTR = nb_types/math.sqrt(2*nb_tokens)
    RootTTR = nb_types/math.sqrt(nb_tokens)
    BilogTTR = math.log(nb_types)/math.log(nb_tokens)
    MTLD = measureOfTextualLexicalDiversity(doc)

    return [TTR, CorrectedTTR, RootTTR, BilogTTR, MTLD]



#---------------------------------------------------------------------
# POS Tag Density features: fr & eng
#---------------------------------------------------------------------
# Those features are taken from article
# paper: Automated assessment of non-native learner essays: Investigating the role of linguistic features
# authors: Sowmya Vajjala, Iowa State University, USA sowmya@iastate.edu


# based on french tagger from https://huggingface.co/qanastek/pos-french
# based on eng tagger from spacy



#TODO: morphologiser in spacy gibven info about tense used for VERB it can be added to the section of VERB
def POSTagDensity(doc):
    """
    Function for density feature calculation. 

    Parameters:
    doc (frenchDoc class oblect): contains .docFull wich is the list of lists [word, word.lemma, word.pos, word.tag]

    Returns:
    typesTags (dict): dictionary with the keys of feature names and items values of the features for this document.
    """
    # # Load the model
    # model = SequenceTagger.load("qanastek/pos-french")
    # # Create sentence object out of doc
    # sentence = Sentence(doc.text)
    # # Predict tags
    # prediction = model.predict(sentence)
    # # Split tagged object to get list of pairs word-tag
    # taggedText = sentence.to_tagged_string().split("→")[1].split(",")

    # List of features to be extracted

    listOfFeatures = ['TotalWords', 'numAdj', 'numNouns', 'numVerbs', 'numPronouns', 'numConjunct', 'numProperNouns', 'numPrepositions', 'numAdverbs', 'numLexicals', 'numInterjections', 'perpronouns' , 'numFunctionWords',  'numDeterminers',  'numVB', 'numVBN', 'numVerbsOnly']
    #'perpronouns' : adding num Personal Pronouns with the hypothesis that they will occur more in Simple Sentences
    #'whperpronouns' : adding num Wh personal pronouns with the hypothesis that they will occur more in Normal sentences.
    #'numFunctionWords' : by Wiki: Articles, Pronouns, Conjunctions, Interjections, Prep, Adverbs, Aux - Verbs.
    #'numTenses' :  Number of different tenses in the sentence.Intuition: More tenses, more difficult to understand
    uniqueVerbs = []

    typesTags = dict.fromkeys(listOfFeatures, 0)

    for words in doc.docFull:
        # words = words.replace(' ', '')
        # words = words.replace('\"', '')
        # words = words.replace('\\', '')
        # # words.replace(str(\), '')
        # words =  words.replace('[', '')
        # words = words.split("/")
        # token_text = words[0]
        # token_tag = words[1]

        token_text = words.text
        token_tag = words.tag_

        # #print(token_text, " ", token_tag)
        if (token_tag== "PPER1S") or (token_tag== "PPER2S") or (token_tag== "PPER3MS") or (token_tag=="PPER3MP")or (token_tag=="PPER3FS") or (token_tag=="PPER3FP"):
            typesTags['numPronouns']  += 1
            typesTags['perpronouns'] += 1
            # if (token_tag=="WP"):
            #     typesTags['whperpronouns'] += 1
            typesTags['numFunctionWords'] += 1
            typesTags['TotalWords'] += 1
        if (token_tag== "VERB") or (token_tag== "VPPMS") or (token_tag== "VPPMP") or (token_tag== "VPPFS") or (token_tag== "VPPFP"):
            typesTags['numVerbs'] += 1
            typesTags['TotalWords'] += 1
            if not token_text in uniqueVerbs:
                uniqueVerbs.append(token_text)
                #TODO: if it has to calculate how many words were used only once then it is not working like that cause if the word is used the second time we are not updating this list to delete this word
            if (token_tag=="VERB"):
                typesTags['numVB'] += 1
            elif(token_tag== "VPPMS") or (token_tag== "VPPMP") or (token_tag== "VPPFS") or (token_tag== "VPPFP"):
                typesTags['numVBN'] += 1
        if (token_tag=="ADJ") or (token_tag== "ADJMS") or (token_tag=="ADJMP") or (token_tag=="ADJFS") or (token_tag=="ADJFP"):
            typesTags['numAdj'] += 1
            typesTags['TotalWords'] += 1
        if (token_tag== "ADV") or (token_tag=="PART"):
            typesTags['numAdverbs'] += 1
            typesTags['numFunctionWords'] += 1
            typesTags['TotalWords'] += 1
        if (token_tag=="PREP") or (token_tag=="COSUB"):
            typesTags['numPrepositions'] += 1
            typesTags['numFunctionWords'] += 1
            typesTags['TotalWords'] += 1
        if (token_tag=="INTJ"):
            typesTags['numInterjections'] += 1
            typesTags['numFunctionWords'] += 1
            typesTags['TotalWords'] += 1
        if (token_tag=="COCO"):
            typesTags['numConjunct'] += 1
            typesTags['numFunctionWords'] += 1
            typesTags['TotalWords'] += 1
        if (token_tag=="NOUN") or (token_tag=="NMS") or (token_tag=="NMP") or (token_tag=="NFS") or (token_tag=="NFP"):
            typesTags['numNouns'] += 1
            typesTags['TotalWords'] += 1
        if (token_tag=="PROPN"):
            typesTags['numProperNouns'] += 1
            typesTags['TotalWords'] += 1
        # if (token_tag=="MD"):
        #     typesTags['numModals'] += 1
        #     typesTags['numauxverbs'] += 1
        #     typesTags['numFunctionWords'] += 1
        #     typesTags['TotalWords'] += 1
        if (token_tag=="DET")  or (token_tag=="DETMS")  or (token_tag=="DETFS"):
            typesTags['numFunctionWords'] += 1
            typesTags['numDeterminers'] += 1
            typesTags['TotalWords'] += 1
            
    typesTags['numLexicals'] += typesTags['numAdj'] + typesTags['numNouns'] + typesTags['numVerbs'] + typesTags['numAdverbs'] + typesTags['numProperNouns'] #Lex.Den = NumLexicals/TotalWords
    typesTags['numVerbsOnly'] = typesTags['numVerbs'] #- typesTags['numauxverbs']

    listOfPOSTagFeatures = ['POS_numNouns', 'POS_numProperNouns', 'POS_numPronouns', 'POS_numConjunct', 'POS_numAdjectives', 'POS_numVerbs',
                            'POS_numAdverbs',  'POS_numModals', 'POS_numPrepositions',  'POS_numInterjections', 'POS_numPerPronouns',
                            'POS_numWhPronouns', 'POS_numLexicals', 'POS_numFunctionWords', 'POS_numDeterminers',
                            'POS_numVerbsVB', 'POS_numVerbsVBG', 'POS_numVerbsVBN', 'POS_numVerbsVBP', 'POS_numVerbsVBZ',
                            'POS_advVar', 'POS_adjVar', 'POS_modVar', 'POS_nounVar', 'POS_verbVar1', 'POS_verbVar2', 'POS_squaredVerbVar1',
                            'POS_correctedVV1']

    posFeatures = dict.fromkeys(listOfPOSTagFeatures,0)

    posFeatures["POS_numNouns"]= (typesTags['numNouns'] + typesTags['numProperNouns']) / typesTags['TotalWords']
    posFeatures["POS_numProperNouns"] = typesTags['numProperNouns'] / typesTags['TotalWords']
    posFeatures["POS_numPronouns"] = typesTags['numPronouns'] / typesTags['TotalWords']
    posFeatures["POS_numConjunct"] = typesTags['numConjunct'] / typesTags['TotalWords']
    posFeatures["POS_numAdjectives"] = typesTags['numAdj'] / typesTags['TotalWords']
    posFeatures["POS_numVerbs"] = typesTags['numVerbs'] / typesTags['TotalWords']
    posFeatures["POS_numAdverbs"] = typesTags['numAdverbs'] / typesTags['TotalWords']
    
    
    # posFeatures["POS_numModals"] =  typesTags['numModals'] / typesTags['TotalWords']
    posFeatures["POS_numPrepositions"] = typesTags['numPrepositions'] / typesTags['TotalWords']
    posFeatures["POS_numInterjections"] = typesTags['numInterjections'] / typesTags['TotalWords']
    posFeatures["POS_numPerPronouns"] = typesTags['perpronouns'] / typesTags['TotalWords']
    # posFeatures["POS_numWhPronouns"] = typesTags['whperpronouns'] / typesTags['TotalWords']
    posFeatures["POS_numLexicals"] = typesTags['numLexicals'] / typesTags['TotalWords'] #Lexical Density
    posFeatures["POS_numFunctionWords"] = typesTags['numFunctionWords'] / typesTags['TotalWords']
    posFeatures["POS_numDeterminers"] = typesTags['numDeterminers'] / typesTags['TotalWords']
    posFeatures["POS_numVerbsVB"] = typesTags['numVB'] / typesTags['TotalWords']
    # posFeatures["POS_numVerbsVBD"] = typesTags['numVBD'] / typesTags['TotalWords']
    
    
    # posFeatures["POS_numVerbsVBG"] =  typesTags['numVBG'] / typesTags['TotalWords']
    posFeatures["POS_numVerbsVBN"] =  typesTags['numVBN'] / typesTags['TotalWords']
    # posFeatures["POS_numVerbsVBP"] = typesTags['numVBP'] / typesTags['TotalWords']
    # posFeatures["POS_numVerbsVBZ"] = typesTags['numVBZ'] / typesTags['TotalWords']
    posFeatures["POS_advVar"] = typesTags['numAdverbs'] / typesTags['numLexicals']
    posFeatures["POS_adjVar"] = typesTags['numAdj'] / typesTags['numLexicals']
    posFeatures["POS_modVar"] = (typesTags['numAdj'] + typesTags['numAdverbs']) / typesTags['numLexicals']
    posFeatures["POS_nounVar"] = (typesTags['numNouns'] + typesTags['numProperNouns']) / typesTags['numLexicals']
    posFeatures["POS_verbVar1"] = typesTags['numVerbsOnly'] / len(uniqueVerbs)# VV1
    posFeatures["POS_verbVar2"] = typesTags['numVerbsOnly'] / typesTags['numLexicals'] # VV2
    posFeatures["POS_squaredVerbVar1"] = (typesTags['numVerbsOnly'] * typesTags['numVerbsOnly']) / len(uniqueVerbs) # VV1
    posFeatures["POS_correctedVV1"] = typesTags['numVerbsOnly'] / math.sqrt(2.0 * len(uniqueVerbs)) # CVV1
    
    #print(posFeatures)
    return posFeatures



def POSTagForEng(doc):
    """
    Function for builging mapping between POSTag notations in french and english tagging. 

    Parameters:
    doc (frenchDoc class oblect): contains .docFull wich is the list of lists [word, word.lemma, word.pos, word.tag]

    Returns:
    typesTags (dict): dictionary with the keys of feature names and items values of the features for this document.
    """
    listOfFeatures = ['TotalWords', 'numAdj', 'numNouns', 'numVerbs', 'numPronouns', 'numConjunct', 'numProperNouns', 'numPrepositions', 'numAdverbs', 'numLexicals', 'numModals', 'numInterjections', 'perpronouns' , 'whperpronouns', 'numauxverbs', 'numFunctionWords',  'numDeterminers', 'numTenses',  'numVB', 'numVBD','numVBG', 'numVBN', 'numVBP', 'numVBZ']
    #'perpronouns' : adding num Personal Pronouns with the hypothesis that they will occur more in Simple Sentences
    #'whperpronouns' : adding num Wh personal pronouns with the hypothesis that they will occur more in Normal sentences.
    #'numFunctionWords' : by Wiki: Articles, Pronouns, Conjunctions, Interjections, Prep, Adverbs, Aux - Verbs.
    #'numTenses' :  Number of different tenses in the sentence.Intuition: More tenses, more difficult to understand
    uniqueVerbs = []

    typesTags = dict.fromkeys(listOfFeatures, 0)
    for token in doc:
        #print(token.text, " ", token.tag_)
        if (token.tag_ == "PRP") or (token.tag_ == "PRP$") or (token.tag_ == "WP") or (token.tag_ =="WP$"):
            typesTags['numPronouns']  += 1
            if (token.tag_ =="PRP"):
                typesTags['perpronouns'] += 1
            if (token.tag_ =="WP"):
                typesTags['whperpronouns'] += 1
            typesTags['numFunctionWords'] += 1
            typesTags['TotalWords'] += 1
        if (token.tag_ == "VB") or (token.tag_ == "VBD") or (token.tag_ == "VBG") or (token.tag_ == "VBN") or (token.tag_ == "VBP") or (token.tag_ == "VBZ"):
            typesTags['numVerbs'] += 1
            typesTags['TotalWords'] += 1
            if not token.text in uniqueVerbs:
                uniqueVerbs.append(token.text)
            if (token.tag_ =="VB"):
                typesTags['numVB'] += 1
            elif (token.tag_ =="VBD"):
                typesTags['numVBD'] += 1
            elif (token.tag_ =="VBG"):
                typesTags['numVBG'] += 1
            elif (token.tag_ =="VBN"):
                typesTags['numVBN'] += 1
            if (token.tag_ =="VBP"):
                typesTags['numVBP'] += 1
            if (token.tag_ =="VBZ"):
                typesTags['numVBZ'] += 1
        if (token.tag_ =="JJ") or (token.tag_ == "JJR") or (token.tag_ =="JJS"):
            typesTags['numAdj'] += 1
            typesTags['TotalWords'] += 1
            if (token.tag_ == "RB") or (token.tag_ =="RBR") or (token.tag_ =="RBS") or (token.tag_ =="RP"):
                typesTags['numAdverbs'] += 1
                typesTags['numFunctionWords'] += 1
                typesTags['TotalWords'] += 1
            if (token.tag_ =="IN"):
                typesTags['numPrepositions'] += 1
                typesTags['numFunctionWords'] += 1
                typesTags['TotalWords'] += 1
            if (token.tag_ =="UH"):
                typesTags['numInterjections'] += 1
                typesTags['numFunctionWords'] += 1
                typesTags['TotalWords'] += 1
            if (token.tag_ =="CC"):
                typesTags['numConjunct'] += 1
                typesTags['numFunctionWords'] += 1
                typesTags['TotalWords'] += 1
            if (token.tag_ =="NN") or (token.tag_ =="NNS"):
                typesTags['numNouns'] += 1
                typesTags['TotalWords'] += 1
            if (token.tag_ =="NNP") or (token.tag_ =="NNPS"):
                typesTags['numProperNouns'] += 1
                typesTags['TotalWords'] += 1
            if (token.tag_ =="MD"):
                typesTags['numModals'] += 1
                typesTags['numauxverbs'] += 1
                typesTags['numFunctionWords'] += 1
                typesTags['TotalWords'] += 1
            if (token.tag_ =="DT"):
                typesTags['numFunctionWords'] += 1
                typesTags['numDeterminers'] += 1
                typesTags['TotalWords'] += 1

    typesTags['numLexicals'] += typesTags['numAdj']+typesTags['numNouns']+typesTags['numVerbs']+typesTags['numAdverbs']+ typesTags['numProperNouns'] #Lex.Den = NumLexicals/TotalWords
    typesTags['numVerbsOnly'] = numVerbs-numauxverbs
    # #print(typesTags)

    return typesTags

#---------------------------------------------------------------------
# Discourse features 
#---------------------------------------------------------------------
# Those features are taken from article
# paper: Automated assessment of non-native learner essays: Investigating the role of linguistic features
# authors: Sowmya Vajjala, Iowa State University, USA sowmya@iastate.edu

# Code was taken from repository of: https://bitbucket.org/nishkalavallabhi/ijaiedpapercode/src/master/features/ContentOverlapFeatures.java
# Code was adapted fro python and fremch language using fr tagger

# TODO: because og automatic annotation there is no points thus no separation on sentences

def getOverlapFeatures(doc):
    """
    Function extracts overlap features. 

    Parameters:
    doc (frenchDoc class oblect): contains .docFull wich is the list of lists [word, word.lemma, word.pos, word.tag]

    Returns:
    overlapFeatures (dict): dictionary with the keys of feature names and items values of the features for this document.
    """
    # doc is given in format of [[list of (text, lemma, pos, tag)], ..., []]
    sentences = doc

    listOfFeatures = ['localNounOverlapCount', 'localArgumentOverlapCount', 'localStemOverlapCount', 'localContentWordOverlap', 'globalNounOverlapCount', 'globalArgumentOverlapCount', 'globalStemOverlapCount', 'globalContentWordOverlap']
    overlapFeatures = dict.fromkeys(listOfFeatures, 0)

    totalSentencesSize = len(doc) #denominator for local overlap features.
    # int totalPairsCount = 0; # Used as the denominator for global overlap features.

    for i in range(0, totalSentencesSize):
        for j in range(i+1, totalSentencesSize):
            sentence1 = sentences[i]
            sentence2 = sentences[j]
            if (isThereNounOverlap(sentence1, sentence2)):
                if (j - i == 1):
                    overlapFeatures['localNounOverlapCount'] += 1
                    overlapFeatures['localArgumentOverlapCount'] += 1
                    overlapFeatures['localStemOverlapCount'] += 1
                overlapFeatures['globalNounOverlapCount'] += 1
                overlapFeatures['globalArgumentOverlapCount'] += 1
                overlapFeatures['globalStemOverlapCount'] += 1
            elif (isThereArgumentOverlap(sentence1, sentence2)):
                if (j-i == 1):
                    overlapFeatures['localArgumentOverlapCount'] += 1
                    overlapFeatures['localStemOverlapCount'] += 1
                overlapFeatures['globalArgumentOverlapCount'] += 1
                overlapFeatures['globalStemOverlapCount'] += 1
            elif (isThereStemOverlap(sentence1, sentence2)):
                if (j-i == 1):
                    overlapFeatures['localStemOverlapCount'] += 1
                overlapFeatures['globalStemOverlapCount'] += 1

            tempContentOverlap = contentWordOverlap(sentence1, sentence2)
            overlapFeatures['globalContentWordOverlap'] += tempContentOverlap
            if (j-i == 1):
                overlapFeatures['localContentWordOverlap'] += tempContentOverlap

    for key in overlapFeatures.keys():
        overlapFeatures[key] = overlapFeatures[key] / totalSentencesSize

    return overlapFeatures


def isThereNounOverlap(sentence1, sentence2):
    for word in sentence1:
        if (word.pos_ == "NOUN") and (word in sentence2):
            return True
    return False

def isThereStemOverlap(sentence1, sentence2):
    if (isThereNounOverlap(sentence1, sentence2) or isThereArgumentOverlap(sentence1, sentence2)):
        return True  # If there is a noun or argument overlap, there is obviously an argument overlap right? No need to test further.
    else:
        for word in sentence1:
            if (not word.pos_ == None):
                lemmaTemp = word.lemma_
                posTemp = word.pos_
                for wordTemp in sentence2:
                    lemmaTemp2 = wordTemp.lemma_
                    posTemp2 = wordTemp.pos_
                    if (lemmaTemp == lemmaTemp2) and ((posTemp2 == "NOUN") or (posTemp == "NOUN") or (posTemp == "PRON")):
                        return True
    return False



def isThereArgumentOverlap(sentence1, sentence2):
    if (isThereNounOverlap(sentence1, sentence2)):
        return True # If there is a noun overlap, there is obviously an argument overlap right? No need to test further.
    # Check for pronoun overlap?
    else:
        for word in sentence1:
            if (word.pos_ == "PRON") and (word in sentence2):
                return True
            else:
                if (word.pos_ == "NOUN") or (word.pos_ == "PRON"):
                    lemmaTemp = word.lemma_
                    posTemp = word.pos_
                    for wordTemp in sentence2:
                        if (lemmaTemp == wordTemp.lemma_) and (posTemp == wordTemp.pos_) and (not posTemp == None):
                            return True

    return False




def contentWordOverlap(sentence1, sentence2):
    overlapsCount = 0
    # Isn't this similar to cosine similarity between content-words?
    for word in sentence1:
        poscatTemp = word.pos_
        lemmaTemp = word.lemma_
        if (not poscatTemp == None) and (not poscatTemp == "PRON"):
            for word2 in sentence2:
                if (lemmaTemp == word2.lemma_):
                    overlapsCount += 1
    return overlapsCount


#---------------------------------------------------------------------
# Chains of references 
#---------------------------------------------------------------------


def getRefExpFeatures(doc):
    """
    Function extracts referencial features. 

    Parameters:
    doc (frenchDoc class oblect): contains .docFull wich is the list of lists [word, word.lemma, word.pos, word.tag]

    Returns:
    refExprFeatures (dict): dictionary with the keys of feature names and items values of the features for this document.
    """
    listOfCounters = ['numWords', 'numSentences', 'numPronouns', 'numPersonalPronouns', 'numPossessivePronouns', 'numDefiniteArticles', 'numProperNouns', 'numNouns']
    refExprCounters = dict.fromkeys(listOfCounters, 0)
    refExprCounters['numSentences'] = len(doc.docPerSent)

    for sentence in doc.docPerSent:
        for wordtagpair in sentence:
            word = wordtagpair.text.lower()
            tag = wordtagpair.tag_
            if (tag != None):
                refExprCounters['numWords'] += 1
                if ((tag == "DET") or (tag == "DETMS") or (tag == "DETFS")) and (word == "the"):
                    refExprCounters['numDefiniteArticles']+= 1
                elif (tag == "PPER1S") or (tag == "PPER2S") or (tag == "PPER3MS") or (tag == "PPER3MP") or (tag == "PPER3FS") or (tag == "PPER3FP"):
                    refExprCounters['numPersonalPronouns']+= 1
                elif (tag == "PRP$"): #TODO: not sure in my tagger this exists but there is Demonstrative Pronoun
                    refExprCounters['numPossessivePronouns']+= 1
                elif (tag == "NOUN") or (tag == "NMS") or (tag == "NMP") or (tag == "NFS") or (tag == "NFP") :
                    refExprCounters['numNouns']+= 1
                if (tag == "PROPN"):
                    refExprCounters['numProperNouns'] += 1

    refExprCounters['numPronouns'] = refExprCounters['numPersonalPronouns'] + refExprCounters['numPossessivePronouns']

    listOfFeatures = ['DISC_RefExprPronounsPerNoun', 'DISC_RefExprPronounsPerSen', 'DISC_RefExprPronounsPerWord','DISC_RefExprPerPronounsPerSen', 'DISC_RefExprPerProPerWord','DISC_RefExprPossProPerSen', 'DISC_RefExprPossProPerWord', 'DISC_RefExprDefArtPerSen','DISC_RefExprDefArtPerWord', 'DISC_RefExprProperNounsPerNoun']
    refExprFeatures = dict.fromkeys(listOfFeatures, 0)

    refExprFeatures["DISC_RefExprPronounsPerNoun"] = refExprCounters['numPronouns'] / refExprCounters['numNouns']
    refExprFeatures["DISC_RefExprPronounsPerSen"] = refExprCounters['numPronouns'] / refExprCounters['numSentences']
    refExprFeatures["DISC_RefExprPronounsPerWord"] = refExprCounters['numPronouns'] / refExprCounters['numWords']
    refExprFeatures["DISC_RefExprPerPronounsPerSen"] = refExprCounters['numPersonalPronouns'] / refExprCounters['numSentences']
    refExprFeatures["DISC_RefExprPerProPerWord"] = refExprCounters['numPersonalPronouns'] / refExprCounters['numWords']
    refExprFeatures["DISC_RefExprPossProPerSen"] = refExprCounters['numPossessivePronouns'] / refExprCounters['numSentences']
    refExprFeatures["DISC_RefExprPossProPerWord"] = refExprCounters['numPossessivePronouns'] / refExprCounters['numWords']
    refExprFeatures["DISC_RefExprDefArtPerSen"] = refExprCounters['numDefiniteArticles'] / refExprCounters['numSentences']
    refExprFeatures["DISC_RefExprDefArtPerWord"]  = refExprCounters['numDefiniteArticles'] / refExprCounters['numWords']
    refExprFeatures["DISC_RefExprProperNounsPerNoun"] = refExprCounters['numProperNouns'] / refExprCounters['numNouns']


    return refExprFeatures



#---------------------------------------------------------------------
# Stuttering 
#---------------------------------------------------------------------

# TODO: problems of french language: 
# 1) reflexive verbs, example: il se sent bien ---> se -- prefix of sent BUT it is not stuttering 
# 2) acticles, example: je sais que un univers ... ---> un -- prefix of univers
# 3) other, example: je sais que quelque ... ---> que -- prefix of quelque

def isPrefix(word1, word2):
    if (word1 == word2[:len(word1)]):
        return True
    else:
        return False

def hasStutterPrefix(word):
    for i in range(2,len(word) - 1):
        #print(word[:i], ' ', word[i:])
        if word[:i] in word[i:]:
            return True
         
    
    return False


def stutterFeatures(doc):
    """
    Function extracts stutter features. 

    Parameters:
    doc (frenchDoc class oblect): contains .docFull wich is the list of lists [word, word.lemma, word.pos, word.tag]

    Returns:
    stutterFeatures (dict): dictionary with the keys of feature names and items values of the features for this document.
    """
    #print(doc.text)
    #print([token.tag_ for token in doc.docFull])

    listOfFeatures = ['stutterToSent', 'stutterToWords', 'stutterStart', 'stutterMiddle', 'stutterEnd', 'stutterNum', 'stutterFirstThird', 'stutterSecondThird', 'stutterThirdThird']
    stutterFeatures = dict.fromkeys(listOfFeatures, 0)

    docLen = len(doc.docFull)
    docThird = round(docLen/2)
    curToken = 0

    
    for sent in doc.docPerSent:
        sentLen = len(sent)
        thirdLen = round(sentLen/2)
        for i, token in enumerate(sent):
            # #print(token, i)
            if (i < len(sent)-1):
                if isPrefix(token.text.lower(), sent[i+1].text.lower()) or (token.tag_ == "PUNCT" and len(token.text) >= 1 and hasStutterPrefix(token.text.lower())):
                    #print("is prefix:  ", isPrefix(token.text.lower(), sent[i+1].text.lower()), "hasStutter: ", hasStutterPrefix(token.text.lower()), '\n' )
                    #print("currentToken: ", token.text, "nextOne: ", sent[i+1].text , '\n')
                    
                    if (i <= thirdLen):
                        stutterFeatures['stutterStart'] += 1
                    elif (thirdLen < i) and (i <= 2*thirdLen):
                        stutterFeatures['stutterMiddle'] += 1
                    else:
                        stutterFeatures['stutterEnd'] += 1



    for i, token in enumerate(doc.docFull):
        if (i < docLen - 1):
            if isPrefix(token.text.lower(), doc.docFull[i+1].text.lower()) or (token.tag_ == "PUNCT" and len(token.text) >= 1 and hasStutterPrefix(token.text.lower())):
                stutterFeatures['stutterNum'] += 1

                #print("is prefix:  ", isPrefix(token.text.lower(), doc.docFull[i+1].text.lower()), "hasStutter: ", hasStutterPrefix(token.text.lower()), '\n' )
                #print("currentToken: ", token.text, "nextOne: ", doc.docFull[i+1].text , '\n')
                if (curToken <= docThird):
                    stutterFeatures['stutterFirstThird'] += 1
                elif (docThird < curToken) and (curToken <= 2*docThird):
                    stutterFeatures['stutterSecondThird'] += 1
                else:
                    stutterFeatures['stutterThirdThird'] += 1
                    

                
    stutterFeatures['stutterToSent'] = stutterFeatures['stutterNum']/len(doc.docPerSent)
    stutterFeatures['stutterToWords'] = stutterFeatures['stutterNum']/docLen
    stutterFeatures['stutterStart'] = stutterFeatures['stutterStart']/len(doc.docPerSent)
    stutterFeatures['stutterMiddle'] = stutterFeatures['stutterMiddle']/len(doc.docPerSent)
    stutterFeatures['stutterEnd'] = stutterFeatures['stutterEnd']/len(doc.docPerSent)
    stutterFeatures['stutterFirstThird'] = stutterFeatures['stutterFirstThird']/docThird
    stutterFeatures['stutterSecondThird'] = stutterFeatures['stutterSecondThird']/docThird
    stutterFeatures['stutterThirdThird'] = stutterFeatures['stutterThirdThird']/docThird

    #print(stutterFeatures)
    return stutterFeatures



#---------------------------------------------------------------------
# Embeddings extraction
#---------------------------------------------------------------------
# based on Doc2Vec pretrained on Wiki 
# TODO: check training data, maybe re-train the model on more apropriative data


def embeddingExtraction(doc):
    # parameters
    model = "../demo/" + dataset + "/model.bin"
    # test_docs="toy_data/test_docs.txt"
    # output_file = "./test_vectors.txt"

    # inference hyper-parameters
    start_alpha = 0.01
    infer_epoch = 1000

    # load Doc2Vec model
    m = g.Doc2Vec.load(model)

    # Build a sample cloud
    sample_cloud = np.array(m.infer_vector(doc.text.split(), alpha=start_alpha, epochs=infer_epoch))

    return sample_cloud




#---------------------------------------------------------------------
# Ratio-based features
#---------------------------------------------------------------------
# TODO: merge this with the new word based features from article 

def extractLinkingRate(doc):
    """
    Function extracts linking rate features. 

    Parameters:
    doc (frenchDoc class oblect): contains .docFull wich is the list of lists [word, word.lemma, word.pos, word.tag]

    Returns:
    conjunctFeatures (dict): dictionary with the keys of feature names and items values of the features for this document.
    """


    listOfFeatures = ['conjunctToSent', 'conjunctTypesToSent', 'conjunctToWords', 'conjunctNum', 'conjunctTypesToTotal', 'conjunctNeighborSent']
    conjunctFeatures = dict.fromkeys(listOfFeatures, 0)

    nbSent = len(doc.docPerSent)
    docLen = len(doc.docFull)

    conjunction = dict()
    for token in doc.docFull:
        # #print(token.pos_)
        if (token.pos_ == "CONJ") or (token.pos_ == "CCONJ"):
            conjunctFeatures['conjunctNum'] += 1
            x = conjunction.setdefault(token.lemma_)
            if (x != None):
                conjunction[token.lemma_] += 1
            else:
                conjunction[token.lemma_] = 0
    
   
    for curSent, sent in enumerate(doc.docPerSent):
        if (curSent < nbSent - 1):
            if (sent[0].text.lower() == doc.docPerSent[curSent + 1][0].text.lower()):
                conjunctFeatures['conjunctNeighborSent'] += 1

    if conjunctFeatures['conjunctNum'] > 0:
        conjunctFeatures['conjunctTypesToTotal'] = len(conjunction) / conjunctFeatures['conjunctNum']
    else:
        conjunctFeatures['conjunctTypesToTotal'] =  0

    conjunctFeatures['conjunctToSent'] = conjunctFeatures['conjunctNum']/nbSent
    conjunctFeatures['conjunctTypesToSent'] = len(conjunction)/nbSent
    conjunctFeatures['conjunctToWords'] = conjunctFeatures['conjunctNum']/docLen
    conjunctFeatures['conjunctNeighborSent'] /= nbSent

    return conjunctFeatures


# TODO: optimise code to not to have doubled code for NOUN & VOERB
# TODO: calculate the time of calculations and check if possible to optimise
# TODO: sameRootRate features 
def extractSynonymsRate(doc):
    """
    Function extracts synonym rate features. 

    Parameters:
    doc (frenchDoc class oblect): contains .docFull wich is the list of lists [word, word.lemma, word.pos, word.tag]

    Returns:
    synonymsFeatures (dict): dictionary with the keys of feature names and items values of the features for this document.
    """

    listOfFeatures = ['synonymToNouns', 'synonymToVerbs', 'averageSynClassNOUN', 'averageSynClassVERB']
    synonymsFeatures = dict.fromkeys(listOfFeatures, 0)

    # dicts for NOUNS
    goupsDict = dict()
    nbOfGroups = 0
    token_dict = dict()
    roots_dict = dict()
    allNouns = dict()

    # dicts for verbs
    goupsDictVERB = dict()
    nbOfGroupsVERB = 0
    token_dictVERB = dict()
    roots_dictVERB = dict()
    allVerbs = dict()
    for token in doc.docFull:
        if (token.pos_ == "NOUN"):
            allNouns.setdefault(token.text.lower(), 1)
        if (token.pos_ == "VERB"):
            allVerbs.setdefault(token.text.lower(), 1)
    nbOfNouns = len(allNouns)
    nbOfVerbs = len(allVerbs)
    for token in doc.docFull:
        if (token.pos_ == "NOUN"):
            if roots_dict.setdefault(token.lemma_, 1) != None :
                roots_dict[token.lemma_] += 1
            if token_dict.setdefault(token.text.lower()) == None :
                bool = False
                for syn in wordnet.synsets(token.text.lower(), lang = 'fra'):
                    for i in syn.lemma_names('fra'):
                        if token_dict.setdefault(i) != None:
                            goupsDict[token_dict[i]] += 1
                            token_dict[token.text.lower()] = token_dict[i]
                            bool = True
                            break
                        else:
                            token_dict.pop(i)
                    if bool == True:
                        break
                else:
                    nbOfGroups += 1
                    goupsDict.setdefault(str(nbOfGroups), 1)
                    token_dict[token.text.lower()] = str(nbOfGroups)

        elif (token.pos_ == "VERB"):
            if roots_dictVERB.setdefault(token.lemma_, 1) != None :
                roots_dictVERB[token.lemma_] += 1
            if token_dictVERB.setdefault(token.text.lower()) == None :
                bool = False
                for syn in wordnet.synsets(token.text.lower(), lang = 'fra'):
                    for i in syn.lemma_names('fra'):
                        if token_dictVERB.setdefault(i) != None:
                            goupsDictVERB[token_dictVERB[i]] += 1
                            token_dictVERB[token.text.lower()] = token_dictVERB[i]
                            bool = True
                            break
                        else:
                            token_dictVERB.pop(i)

                    if bool == True:
                        break
                else:
                    nbOfGroupsVERB += 1
                    goupsDictVERB.setdefault(str(nbOfGroupsVERB), 1)
                    token_dictVERB[token.text.lower()] = str(nbOfGroupsVERB)

    synonymsFeatures['synonymToNouns'] = nbOfGroups / nbOfNouns
    synonymsFeatures['averageSynClassNOUN'] = sum([int(goupsDict[volume]) for volume in goupsDict])/nbOfGroups
    synonymsFeatures['synonymToVerbs'] = nbOfGroupsVERB / nbOfVerbs
    synonymsFeatures['averageSynClassVERB'] = sum([int(goupsDictVERB[volume]) for volume in goupsDictVERB])/nbOfGroupsVERB

    return synonymsFeatures








#---------------------------------------------------------------------
# Extract all features and provide to the TextProcess()
#---------------------------------------------------------------------

def extractTextFeatures(doc):
    """
    Function createing the dictionary of the feature names of the category. 

    Returns:
    categoryDict (dict): dictionary with the keys of category names and items as lists of feature names.
    """
    if (language == 'french'):
        # Linking Rate feature extraction
        features_linking_rate = extractLinkingRate(doc)
        # Synonym Rate feature extraction
        features_synonyms_rate = extractSynonymsRate(doc)
        # Diversity feature extraction
        features_diversity = lexicalDiversity(doc.docFull)
        # Density feature extraction
        features_density = POSTagDensity(doc)
        # Discourse feature extraction
        features_discourse  = getOverlapFeatures(doc.docPerSent)
        # Referencial feature extraction
        features_reference = getRefExpFeatures(doc)
    else:
        print('English version of tagging etraction is not concistent with french one')

    return [features_linking_rate, features_synonyms_rate, features_diversity, features_density, features_reference, features_discourse]



def createFeatureLists():
    """
    Function createing the dictionary of the feature names of the category. 

    Returns:
    categoryDict (dict): dictionary with the keys of category names and items as lists of feature names.
    """
    ceategoryDict = {'LinkingRate':['category', 'conjunctToSent', 'conjunctTypesToSent', 'conjunctToWords', 'conjunctNum', 'conjunctTypesToTotal', 'conjunctNeighborSent']
                     , 'SynonymRate':['category','synonymToNouns', 'synonymToVerbs', 'averageSynClassNOUN', 'averageSynClassVERB']
                     , 'featureEmbedName':['category'] + ['emb_axis_' + str(i) for i in range(100)]
                     , 'Diversity':['category','TTR', 'CorrectedTTR', 'RootTTR', 'BilogTTR', 'MTLD']
                     , 'Density':['category','POS_numNouns', 'POS_numProperNouns', 'POS_numPronouns', 'POS_numConjunct', 'POS_numAdjectives', 'POS_numVerbs',
                            'POS_numAdverbs',  'POS_numModals', 'POS_numPrepositions',  'POS_numInterjections', 'POS_numPerPronouns',
                            'POS_numWhPronouns', 'POS_numLexicals', 'POS_numFunctionWords', 'POS_numDeterminers',
                            'POS_numVerbsVB', 'POS_numVerbsVBG', 'POS_numVerbsVBN', 'POS_numVerbsVBP', 'POS_numVerbsVBZ',
                            'POS_advVar', 'POS_adjVar', 'POS_modVar', 'POS_nounVar', 'POS_verbVar1', 'POS_verbVar2', 'POS_squaredVerbVar1',
                            'POS_correctedVV1']
                    , 'Discourse':['category','localNounOverlapCount', 'localArgumentOverlapCount', 'localStemOverlapCount', 'localContentWordOverlap', 'globalNounOverlapCount', 'globalArgumentOverlapCount', 'globalStemOverlapCount', 'globalContentWordOverlap']
                    , 'Reference':['category','DISC_RefExprPronounsPerNoun', 'DISC_RefExprPronounsPerSen', 'DISC_RefExprPronounsPerWord','DISC_RefExprPerPronounsPerSen', 'DISC_RefExprPerProPerWord','DISC_RefExprPossProPerSen', 'DISC_RefExprPossProPerWord', 'DISC_RefExprDefArtPerSen','DISC_RefExprDefArtPerWord', 'DISC_RefExprProperNounsPerNoun']
    }
    return ceategoryDict



def french_to_english_text(text):
    """
    Function cleaning the french letters in the filenames if needed. 

    Parameters:
    text (str): Name of the file/name of the directory.
    
    Returns:
    translated_text (str): cleaned name of directory name.
    """
    french_to_english = {'é': 'e', 'á': 'a','à': 'a','â': 'a','ç': 'c','é': 'e','è': 'e','ê': 'e','ë': 'e','î': 'i','ï': 'i','ô': 'o', 'ù': 'u','û': 'u','ç': 'c',}
    translated_text = ''
    for char in text:
        translated_text += french_to_english.get(char, char)
    return translated_text

def sanitize_filename(filename):
    """
    Function cleaning the filenames from special symbols if needed. 

    Parameters:
    filename (str): Name of the file/name of the directory.
    
    Returns:
    (str): cleaned name of directory name.
    """
    # Example: Remove spaces and special characters
    return filename.replace(" ", "").replace("-", "").replace("_", "").replace(",", "")



def TextProcess():
    """
    Main function that reads .txt files from the directory with transcripts and 
    executes feature extraction. Transcript directory should contain subfolders: 
                            full, beg, mid, end
    Transcripts are named with the ID in the dataset. 

    Parameters:
    rootDirPath (str): Root directory
    dataset: Name of the dataset
    
    Returns:
    str(featureCategories[idx])+".csv" (DataFrame): array of N x nb_cat_feat dimensions where
                                            N -- number of data samples
                                            nb_cat_feat -- number of features in the {category}
                                            category: linkingRate, devirsity, density, discourse, 
                                            reference, synonymRate
    """
    # Data directory
    data_dir=os.path.join(rootDirPath, "data", dataset)
    # Transcript directory
    transcript_dir=os.path.join(data_dir, "transcripts")
    # Create the list of clips interesting for the analysis
    if clip == "all":
        subdir_list=['full', 'beg', 'mid', 'end']
    else:
        subdir_list=[str(clip)]
    # Go through the subdirectories to read the .txt files and extract features
    for subdir in subdir_list: 
        # Create a psth of subdirectory subtaining the .txt files corresponding to the clip
        subdir_path = os.path.join(transcript_dir, subdir)
        # Create the dictionary of the features of each category
        ceategoryDict = createFeatureLists()
        # Go throug the directory {rootDir}/data/{dataset}/transcripts/{subdir}
        for file in os.listdir(subdir_path):
            # Check whether the found file is .txt
            if file.endswith('.txt'):
                # Remove .txt extension and sub-directory prefix to get ID of the file 
                cleaned_name = file.replace(".txt", "")
                # Create POSTagging for the .txt file
                doc = text2sentence(transcript_dir, file)
                # Extract textual features. This outputs the list of lists. 
                # Each list corresponds to the feature category and aligned with ceategoryDict and featureCategories.
                featureLists = extractTextFeatures(doc)
                # Go through each category list
                for idx, features_list in enumerate(featureLists):
                    # Create the new line for the feature .csv file out of the features extracted from the new document
                    new_data_row = [cleaned_name] + list(features_list.values())
                    # Data directory for saving textual features
                    feature_file=os.path.join(data_dir, "lingustic_features", str(featureCategories[idx])+".csv")
                    # Try to open .csv with features of the considered category 
                    try:
                        df_features = pd.read_csv(feature_file, index_col=0, names=features_list)
                    except FileNotFoundError:
                        # Otherwise create the new dataset
                        df_features = pd.DataFrame()
                    # Add and save the new line to the features of the category 
                    df_features = df_features._append(pd.DataFrame([new_data_row], columns=ceategoryDict[str(featureCategories[idx])], index=cleaned_name))
                    df_features.to_csv(feature_file, mode='w')
                    



def readMeta():
    # Data directory
    data_dir=os.path.join(rootDirPath, "data", dataset, dataset+"_meta_data.csv")
    # Read metadata
    meta_data= pd.read_csv(data_dir, sep=";")
    # Take language from metadata
    language = meta_data['language']

    
if __name__ == "__main__":

    # Functionality: Punkt is a pre-trained tokenizer that helps in dividing 
    #                a text into a list of sentences or words. It's crucial for tasks that 
    #                require text preprocessing, such as sentence segmentation and word tokenization.
    nltk.download('punkt')

    # Functionality: WordNet is a large lexical database of English that groups words into 
    #               sets of synonyms called synsets. It provides short definitions and usage 
    #               examples, and it is widely used for tasks like word sense disambiguation, 
    #               text analysis, and natural language understanding.
    nltk.download('wordnet')

    # Functionality: The Open Multilingual WordNet provides translations of the English WordNet
    #                synsets into various other languages. This resource is useful for 
    #                multilingual natural language processing tasks and for applications that 
    #                require word sense alignment across different languages.
    nltk.download('omw-1.4')

    # Fix the list of categories of features
    featureCategories=['LinkingRate', 'SynonymRate', 'Diversity', 'Density', 'Reference', 'Discourse']
    
    # Read the configuration
    config = read_config()

    # Use the configuration values
    rootDirPath = config['rootDirPath']
    dataset = config['dataset']
    dimension = config['dimension']
    clip = config['clip']

    # Example usage in your script
    print(f"Root Directory Path: {rootDirPath}")
    print(f"Dataset: {dataset}")
    print(f"Dimension: {dimension}")
    print(f"Clip: {clip}")

    # Adding the rootDirPath to the system path
    sys.path.append(rootDirPath)
    # Read meta information about the dataset
    readMeta()
    TextProcess()

