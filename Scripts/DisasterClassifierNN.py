import pandas as pd
import re
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class DisasterClassifierNN:
    def __init__(self, train_path, test_path, submission_file_path, vocab_size=10000, max_length=50):
        self.train_path = train_path
        self.test_path = test_path
        self.submission_file_path = submission_file_path
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.model = self._build_model()
        self.tokenizer = None

    def load_data(self):
        train = pd.read_csv(self.train_path)
        test = pd.read_csv(self.test_path)
        return train, test

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

    def prepare_data(self, train, test):
        train['text'] = train['text'].apply(self.clean_text)
        test['text'] = test['text'].apply(self.clean_text)

        X_train, X_test, y_train, y_test = train_test_split(train['text'], train['target'], test_size=0.2, random_state=42)

        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(X_train)

        X_train_seq = self.tokenizer.texts_to_sequences(X_train)
        X_train_pad = pad_sequences(X_train_seq, maxlen=self.max_length, padding='post', truncating='post')

        X_test_seq = self.tokenizer.texts_to_sequences(X_test)
        X_test_pad = pad_sequences(X_test_seq, maxlen=self.max_length, padding='post', truncating='post')

        return X_train_pad, y_train, X_test_pad, y_test

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size, 32, input_length=self.max_length),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train_model(self, X_train, y_train, X_test, y_test, num_epochs=10):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        history = self.model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test), verbose=2,
                                 callbacks=[early_stopping])

    def make_predictions(self, test_data):
        X_val_seq = self.tokenizer.texts_to_sequences(test_data['text'])
        X_val_pad = pad_sequences(X_val_seq, maxlen=self.max_length, padding='post', truncating='post')
        predictions = self.model.predict(X_val_pad)
        binary_predictions = (predictions > 0.5).astype(int)
        return binary_predictions
    
    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        binary_predictions = (y_pred > 0.5).astype(int)
        f1 = f1_score(y_test, binary_predictions)
        print(f"F1 Score on test set: {f1}")
        return f1

    def generate_submission(self, predictions):
        sample_submission = pd.read_csv(self.submission_file_path)
        sample_submission["target"] = predictions
        sample_submission.to_csv("Data/Output/nn_submission.csv", index=False)

    def run_disaster_classifier_nn(self):
        train, test = disaster_classifier.load_data()
        X_train, y_train, X_test, y_test = disaster_classifier.prepare_data(train, test)
        disaster_classifier.train_model(X_train, y_train, X_test, y_test)
        disaster_classifier.evaluate_model(X_test, y_test)
        predictions = disaster_classifier.make_predictions(test)
        disaster_classifier.generate_submission(predictions)

if __name__ == "__main__":
    disaster_classifier = DisasterClassifierNN('Data/Input/train.csv', 'Data/Input/test.csv', 'Data/Input/sample_submission.csv')
    disaster_classifier.run_disaster_classifier_nn()