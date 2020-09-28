# MLP for the IMDB problem
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

def CreateModel():
    # load the dataset but only keep the top n words, zero the rest
    top_words = 5000
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

    max_words = 500
    X_train = sequence.pad_sequences(X_train, maxlen=max_words)
    X_test = sequence.pad_sequences(X_test, maxlen=max_words)

    model = object()
    print('MODELS AVAILABLE')
    print('1. MNB \n 2. SVM \n 3. Random Forest \n 4. CNN')
    slctd_mdl = input('SELECT THE MODEL YOU WANT TO USE')

    if (slctd_mdl == '1'):
        # import and instantiate MultinomialNB
        from sklearn.naive_bayes import MultinomialNB

        nb = MultinomialNB(alpha='0.0001', fit_prior=True)

        # fit a Multinomial Naive Bayes model
        nb.fit(X_train, y_train)

        # make class predictions
        y_pred_class = nb.predict(X_test)

        # generate classification report
        from sklearn import metrics

        print(metrics.classification_report(y_test, y_pred_class))

        model = nb
    if (slctd_mdl == '2'):
        # Import svm model
        from sklearn import svm

        # Create a svm Classifier
        clf = svm.SVC()  # Linear Kernel

        # Train the model using the training sets
        clf.fit(X_train, y_train)

        # Predict the response for test dataset
        y_pred = clf.predict(X_test)

        # generate classification report
        from sklearn import metrics

        print(metrics.classification_report(y_test, y_pred))

        model = clf
    if (slctd_mdl == '3'):
        # Import rf model
        from sklearn.ensemble import RandomForestClassifier

        # Create a Gaussian Classifier
        rf = RandomForestClassifier(n_estimators=100)

        # Train the model using the training sets
        rf.fit(X_train, y_train)

        # Predict the response for test dataset
        y_pred = rf.predict(X_test)

        # generate classification report
        from sklearn import metrics

        print(metrics.classification_report(y_test, y_pred))

        model = rf

    elif (slctd_mdl == '4'):
        # create the model
        model = Sequential()
        model.add(Embedding(top_words, 32, input_length=max_words))
        model.add(Flatten())
        model.add(Dense(250, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        # Fit the model
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=128)
    else:
        print('Select appropriate model')

