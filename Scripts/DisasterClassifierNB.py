import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

class DisasterClassifierNB:
    def __init__(self, train_data_path, test_data_path):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.train = None
        self.test = None
        self.tfidf = None
        self.model = None

    def load_data(self):
        self.train = pd.read_csv(self.train_data_path)
        self.test = pd.read_csv(self.test_data_path)

    # References:
    # https://monkeylearn.com/blog/text-cleaning/
    # https://stackoverflow.com/a/47091370
    # https://stackoverflow.com/a/49146722
    def clean_text(self, text):
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub(r"won\'t", " will not", text)
        text = re.sub(r"won\'t've", " will not have", text)
        text = re.sub(r"can\'t", " can not", text)
        text = re.sub(r"don\'t", " do not", text)    
        text = re.sub(r"can\'t've", " can not have", text)
        text = re.sub(r"ma\'am", " madam", text)
        text = re.sub(r"let\'s", " let us", text)
        text = re.sub(r"ain\'t", " am not", text)
        text = re.sub(r"shan\'t", " shall not", text)
        text = re.sub(r"sha\n't", " shall not", text)
        text = re.sub(r"o\'clock", " of the clock", text)
        text = re.sub(r"y\'all", " you all", text)
        text = re.sub(r"n\'t", " not", text)
        text = re.sub(r"n\'t've", " not have", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'s", " is", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'d've", " would have", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'ll've", " will have", text)
        text = re.sub(r"\'t", " not", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'m", " am", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r'<.*?>',' ',text) 
        text = re.sub(r'[0-9]', '', text)
        text = re.sub("["
                            u"\U0001F600-\U0001F64F"  # removal of emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            "]+",' ',text)
        
        text = re.sub('[^a-zA-Z]',' ',text) 
        text = re.sub(r"\([^()]*\)", "", text)
        text = re.sub('@\S+', '', text)
        text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), '', text)
        text = text.lower()
        return text

    # Reference: https://stackoverflow.com/a/5486535
    def text_preprocessing(self, text):
        tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
        cleaned_text = self.clean_text(text)
        tokenized_text = tokenizer.tokenize(cleaned_text)
        filtered_text = [w for w in tokenized_text if w not in stopwords.words('english')]
        combined_text = ' '.join(filtered_text)
        return combined_text

    def prepare_data(self):
        self.load_data()
        self.train['text'] = self.train['text'].apply(self.text_preprocessing)
        self.test['text'] = self.test['text'].apply(self.text_preprocessing)
        self.tfidf = TfidfVectorizer(min_df=1, max_df=0.5, ngram_range=(1, 2), sublinear_tf=True)
        train_tfidf = self.tfidf.fit_transform(self.train['text'])
        test_tfidf = self.tfidf.transform(self.test["text"])
        X_train, X_test, y_train, y_test = train_test_split(train_tfidf, self.train["target"], test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test, test_tfidf

    def train_model(self, X_train, y_train, alpha=0.8):
        self.model = MultinomialNB(alpha=alpha)
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        y_pred_test = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred_test)
        precision = precision_score(y_test, y_pred_test)
        recall = recall_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test)

        print(f"Accuracy on test set: {accuracy:.4f}")
        print(f"Precision on test set: {precision:.4f}")
        print(f"Recall on test set: {recall:.4f}")
        print(f"F1 Score on test set: {f1:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_test))

    def generate_submission(self, submission_file_path, predictions):
        sample_submission = pd.read_csv(submission_file_path)
        sample_submission["target"] = predictions
        sample_submission.to_csv("Data/Output/nb_submission.csv", index=False)

    def run_disaster_classifier_nb(self):
        X_train, X_test, y_train, y_test, test_tfidf = self.prepare_data()
        self.train_model(X_train, y_train)
        self.evaluate_model(X_test, y_test)
        predictions = self.model.predict(test_tfidf)
        submission_file_path = "Data/Input/sample_submission.csv"
        self.generate_submission(submission_file_path, predictions)

if __name__ == "__main__":
    model = DisasterClassifierNB('Data/Input/train.csv', 'Data/Input/test.csv')
    model.run_disaster_classifier_nb()