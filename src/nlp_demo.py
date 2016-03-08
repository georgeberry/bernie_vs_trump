####This is a simple example to show what the different tools do



    import string
    import nltk


    T = "I would like to offer my sincerest condolences to the Scalia family after the passing of Justice Scalia. Justice Scalia was a remarkable person and a brilliant Supreme Court Justice, one of the best of all time. His career was defined by his reverence for the Constitution and his legacy of protecting Americans’ most cherished freedoms. He was a Justice who did not believe in legislating from the bench and he is a person whom I held in the highest regard and will always greatly respect his intelligence and conviction to uphold the Constitution of our country. My thoughts and prayers are with his family during this time
    B = "DENVER – U.S. Sen. Bernie Sanders issued the following statement on Saturday on the passing of U.S. Supreme Court Justice Antonin Scalia.While I differed with Justice Scalia’s views and jurisprudence, he was a brilliant, colorful and outspoken member of the Supreme Court. My thoughts and prayers are with his family and his colleagues on the court who mourn his passing.”


def Tokenizer(sentence):
    #Coverts uppercase to lower
    sentence = sentence.lower()
    #Removes all punctuation
    sentence = sentence.translate(string.maketrans("",""), string.punctuation)
    #Converts sentence to tokens
    tokens = nltk.word_tokenize(sentence)
    #Removes stopword
    words = [word for word in tokens if not word in stopwords]
    #Stems the tokens using the PorterStemmer
    stems = [stemmer.stem(word) for word in words]
    return stems