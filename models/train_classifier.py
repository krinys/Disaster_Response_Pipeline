import sys
import sqlite3
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from nltk import pos_tag,ne_chunk
from sklearn.metrics import confusion_matrix
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,TfidfVectorizer
nltk.download('words')
nltk.download('punkt')
nltk.download('average_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    conn=sqlite3.connect(database_filepath)
    df = pd.read_sql('SELECT * FROM information ',con=conn)
    category_names=df.drop(columns=['id','message','original','genre'],axis=1,inplace=False).columns
    df=df.drop(df[df.related>1].index)
    X =df.message.values
    y = df.drop(columns=['id','message','original','genre'],axis=1,inplace=False).values

    return X,y,category_names

def tokenize(text):
    text=text.lower()# lower the word
    text=re.sub(r"[^a-zA-Z0-9]"," ",text)#remove the punctutaion
    words=word_tokenize(text)#tokenize text
    words=[w for w in words if w not in stopwords.words("english")]# remove the words without enough meaning
    words=[PorterStemmer().stem(w) for w in words]
    words=[WordNetLemmatizer().lemmatize(w) for w in words]
    return words


def build_model():
    forest = RandomForestClassifier(min_samples_split=100,
                                  min_samples_leaf=20,max_depth=8,max_features='sqrt' ,random_state=1)  # 生成随机森林多分类器
    multi_target_forest = MultiOutputClassifier(forest)  # 构建多输出多分类器
    pipeline = Pipeline([
    ('vect',CountVectorizer(tokenizer=tokenize)),
    ('tfidf',TfidfTransformer()),
    ('clf',MultiOutputClassifier(forest))
    ])
    parameters={'clf__estimator__n_estimators': [50, 100, 200]}
    cv = GridSearchCV(pipeline,parameters,n_jobs=-1,cv=2)
    return cv
def evaluate_model(model, X_test, y_test, category_names):
    y_pred=model.predict(X_test)
    scores=[]
    for i in range(0,36,1):
        score=[]
        score.append(recall_score(y_test[:,i],y_pred[:,i]))
        score.append(accuracy_score(y_test[:,i],y_pred[:,i]))
        score.append(precision_score(y_test[:,i],y_pred[:,i]))
        scores.append(score)
    report=pd.DataFrame(scores,index=category_names,columns=['recall','accuracy','precision'])
    for index in category_names:
        print("the accuracy of model is {} in {}".format(report['accuracy'][index],index))
        print("the precision of model is {} in {}".format(report['precision'][index],index))
        print("the recall of model is {} in {}".format(report['recall'][index],index))                               
def save_model(model, model_filepath):
    joblib.dump(model,model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        print("the best number of estimators is {}".format(model.best_params_['clf__estimator__n_estimators']))
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()