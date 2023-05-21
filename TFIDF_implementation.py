import numpy as np
import nltk


class TFIDF:

    def term_frequency(self, term, sentence):
        # please use nltk.word_tokenize
        count = 0 
        sentence = sentence.lower()
        words = nltk.word_tokenize(sentence)
        for word in words:
            if word == term:
                count+=1

        word_dict = {}
        unique_words = np.unique(words)
        for word in unique_words:
            word_dict[word] = 0 
            for w in words:
                if w == word:
                    word_dict[w] += 1
    
        lower = sorted(word_dict.values())[-1]
        
        return 0.5 + 0.5 * (count/lower)

    def inverse_document_frequency(self, term, corpus):
        # please use nltk.word_tokenize
        count = 0 
        for sent in corpus:
            sent = sent.lower()
            words = nltk.word_tokenize(sent)
            for word in words:
                if word == term:
                    count += 1
                    break
        # print(len(corpus))
        return np.log10(len(corpus)/count)



    def tfidf(self, term, sentence, corpus):
        tf = self.term_frequency(term, sentence)
        idf = self.inverse_document_frequency(term, corpus)
        return tf*idf
    
sent1 = "The quick brown fox jumps over the lazy dog"
sent2 = "Never jump over the lazy dog quickly"

corpus = [sent1, sent2]
# 'mama', 'quick The brown dog the. the'

tfidf = TFIDF()




print(tfidf.term_frequency('mama', 'quick The brown dog the. the'))

print("IDF:", tfidf.inverse_document_frequency("quick", corpus))
