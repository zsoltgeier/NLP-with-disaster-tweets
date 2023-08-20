# Import necessary libraries
import streamlit as st
import pandas as pd
import re
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set style
sns.set_palette("Set2")
plt.style.use('ggplot')

# Title and description
st.sidebar.title("by Zsolt Geier")
st.title("Disaster Tweet Analysis with Neural Networks")
st.markdown("This Streamlit app showcases my submission for [Kaggle's Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started) competition. Using a neural network, I analyze tweets to determine whether they are related to real disasters or not.")
# Load and Display Data
st.header("Data Overview")

# Training Data
with st.expander("Training Data"):
    train = pd.read_csv('Data/Input/train.csv')
    st.write(f"Training data shape: {train.shape}")
    st.dataframe(train.head())

# Testing Data
with st.expander("Testing Data"):
    test = pd.read_csv('Data/Input/test.csv')
    st.write(f"Testing data shape: {test.shape}")
    st.dataframe(test.head())

# Check missing values
with st.expander("Missing Values"):
    st.subheader("Training Data")
    missing_train = train.isnull().sum()
    st.write(missing_train)

    st.subheader("Testing Data")
    missing_test = test.isnull().sum()
    st.write(missing_test)

st.caption('The ratio of missing values in test and training sets are around the same, we can assume that they were taken from the same sample.')

# Unique Values in keyword and location columns
with st.expander("Unique Values"):
    st.subheader("Keyword and Location")
    st.write(f'Number of unique values in keyword: {train["keyword"].nunique()} (Training) - {test["keyword"].nunique()} (Test)')
    st.write(f'Number of unique values in location: {train["location"].nunique()} (Training) - {test["location"].nunique()} (Test)')

st.caption('The location column has too many missing and unique values, so I wont use it as a feature.')

# Target distribution in keywords
with st.expander("Target Distribution in keywords"):
    train['target_mean'] = train.groupby('keyword')['target'].transform('mean')
    # Get 20 evenly distributed keywords
    num_keywords = 20
    unique_keywords = train['keyword'].unique()
    step = len(unique_keywords) // num_keywords
    selected_keywords = unique_keywords[::step]

    filtered_train = train[train['keyword'].isin(selected_keywords)]

    fig = plt.figure(figsize=(10, 8))
    ax = sns.countplot(
        y=filtered_train.sort_values(by='target_mean', ascending=False)['keyword'],
        hue=filtered_train.sort_values(by='target_mean', ascending=False)['target']
    )
    plt.xlabel("Count")
    plt.ylabel("Keywords")
    plt.title('Target Distribution in Keywords')
    plt.legend(loc=1)
    st.pyplot(fig)
    train.drop(columns=['target_mean'], inplace=True)

# Overall target distribution and count in the training set
with st.expander("Overall Target Distribution"):
    fig, axes = plt.subplots(ncols=2, figsize=(15, 5))

    train.groupby('target').count()['id'].plot(kind='pie', ax=axes[0], labels=['Not Disaster', 'Disaster'], autopct='%1.1f%%')
    sns.countplot(x=train['target'], hue=train['target'], ax=axes[1])

    axes[0].set_ylabel('')
    axes[1].set_ylabel('')
    axes[1].set_xticklabels(['Not Disaster', 'Disaster'])
    for ax in axes:
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

    axes[0].set_title('Target Distribution in Training Set', fontsize=13)
    axes[1].set_title('Target Count in Training Set', fontsize=13)

    st.pyplot(fig)

with st.expander("Example Tweets"):
    disaster_tweets = train[train['target'] == 1]['text']
    st.write("Example disaster tweet:", disaster_tweets.values[1])

    non_disaster_tweets = train[train['target'] == 0]['text']
    st.write("Example non-disaster tweet:", non_disaster_tweets.values[1])

# Word count in tweets
with st.expander("Word Count in Tweets"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    tweet_len_disaster = train[train['target'] == 1]['text'].str.split().map(lambda x: len(x))
    tweet_len_non_disaster = train[train['target'] == 0]['text'].str.split().map(lambda x: len(x))

    sns.histplot(tweet_len_disaster, ax=ax1, color=sns.color_palette()[1], bins=10) 
    ax1.set_title('Disaster tweets')

    sns.histplot(tweet_len_non_disaster, ax=ax2, color=sns.color_palette()[0], bins=10)
    ax2.set_title('Non-disaster tweets')

    fig.suptitle('Word count in tweets')
    st.pyplot(fig)

st.caption('TODO: Experiment with feature engineering')

# Define function for text cleaning
def clean_text(text):
    #removal of url
    text = re.sub('https?://\S+|www\.\S+', '', text)
    
    #decontraction
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

    #removal of html tags
    text = re.sub(r'<.*?>',' ',text) 
    
    # Match all digits in the string and replace them by empty string
    text = re.sub(r'[0-9]', '', text)
    text = re.sub("["
                           u"\U0001F600-\U0001F64F"  # removal of emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+",' ',text)
    
    # filtering out miscellaneous text.
    text = re.sub('[^a-zA-Z]',' ',text) 
    text = re.sub(r"\([^()]*\)", "", text)

    # remove mentions
    text = re.sub('@\S+', '', text)
    
    # remove punctuations
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), '', text)

    # Lowering all the words in text
    text = text.lower()

    return text

# Apply text cleaning to training and testing data
train['text'] = train['text'].apply(lambda x: clean_text(x))
test['text'] = test['text'].apply(lambda x: clean_text(x))

trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"

# Model Building Section
st.header("Model Building")

with st.form(key="my_form"):
    vocab_size = st.number_input("Vocabulary Size", min_value=100, max_value=10000, value=10000, step=1)
    embedding_dim = st.number_input("Embedding Dimension", min_value=16, max_value=256, value=32, step=1)
    max_length = st.number_input("Max Sequence Length", min_value=10, max_value=100, value=50, step=1)
    num_epochs = st.number_input("Number of Epochs", min_value=1, value=5, step=1)
    user_input_clean = st.text_input("Enter a text for prediction:", "A massive #Earthquake struck the coastal region, causing widespread devastation. 🤕 www.example.com")
    st.caption('Make sure the text is in English! You can see the prediction result and the data preprocessing steps at the \'Model Prediction\' section after training the model.')
    submitted_parameters = st.form_submit_button("Train the model")


if submitted_parameters:    
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Display model summary
    with st.expander("Model summary:"):
        model.summary(print_fn=lambda x: st.text(x))

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(train['text'], train['target'], test_size=0.2, random_state=42)

    # Tokenization and padding
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    # Convert data to NumPy arrays
    X_train_pad = np.array(X_train_pad)
    y_train = np.array(y_train)
    X_test_pad = np.array(X_test_pad)
    y_test = np.array(y_test)

    # Train the model
    history = model.fit(X_train_pad, y_train, epochs=num_epochs, validation_data=(X_test_pad, y_test), verbose=2)
    

    with st.expander("Model Prediction:"):
        user_input = clean_text(user_input_clean)
        user_input_seq = tokenizer.texts_to_sequences([user_input])
        user_input_pad = pad_sequences(user_input_seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)

        user_prediction = model.predict(user_input_pad)

        st.markdown("### Original text:")
        st.write(user_input_clean)
        st.markdown("### Cleaned text:")
        st.write(user_input)
        st.markdown("### Tokenized text:")
        st.write(np.array(user_input_seq))
        st.markdown("### Tokenized and padded text:")
        st.write(user_input_pad)
        st.markdown("### Prediction score:")
        st.write(user_prediction)

        st.markdown("### Prediction Result:")
        if user_prediction > 0.5:
            st.write("**This is a disaster-related text.**")
        else:
            st.write("**This is not a disaster-related text.**")

    # Model Evaluation Section (if applicable)
    with st.expander("Model Evaluation on training and validation sets:"):

        # Display training and validation accuracy and loss plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.set_title('Training and Validation Accuracy')

        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.set_title('Training and Validation Loss')

        st.pyplot(fig)

        # Display other evaluation metrics, confusion matrices, etc.
        # Function to plot confusion matrix with F1 score
        def plot_confusion_matrix_and_f1(y_true, y_pred, class_labels, title):
            # Calculate the confusion matrix
            conf_matrix = confusion_matrix(y_true, y_pred)

            # Calculate the F1 score
            f1 = f1_score(y_true, y_pred)

            # Create a figure and axis
            plt.figure(figsize=(8, 6))
            plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(title + ' -- F1 Score: ' + str(f1))
            plt.colorbar()

            # Add labels to the plot
            tick_marks = np.arange(len(class_labels))
            plt.xticks(tick_marks, class_labels)
            plt.yticks(tick_marks, class_labels)

            # Label the cells with counts
            thresh = conf_matrix.max() / 2.0
            for i in range(conf_matrix.shape[0]):
                for j in range(conf_matrix.shape[1]):
                    plt.text(j, i, format(conf_matrix[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if conf_matrix[i, j] > thresh else "black")

            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.tight_layout()

                
            # Show the plot
            st.pyplot(plt)

        y_pred_train = model.predict(X_train_pad)
        y_pred_train_binary = (y_pred_train > 0.5).astype(int)

        y_pred_test = model.predict(X_test_pad)
        y_pred_test_binary = (y_pred_test > 0.5).astype(int)

        class_labels = ['Negative', 'Positive']

        # Training dataset
        plot_confusion_matrix_and_f1(y_train, y_pred_train_binary, class_labels, 'Confusion Matrix on Training Dataset')
        # Test dataset
        plot_confusion_matrix_and_f1(y_test, y_pred_test_binary, class_labels, 'Confusion Matrix on Test Dataset')