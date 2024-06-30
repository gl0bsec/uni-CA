import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import pandas as pd
import nltk
from collections import Counter
import plotly.graph_objects as go
from nltk.corpus import stopwords
import plotly.express as px
from collections import Counter
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer

def apply_bertopic(df, text_column):
    # Initialize the SentenceTransformer model
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Extract the text data from the DataFrame
    texts = df[text_column].tolist()
    
    # Initialize BERTopic
    topic_model = BERTopic(embedding_model=sentence_model)
    
    # Fit the model on the text data
    topics, probs = topic_model.fit_transform(texts)
    
    # Add the topics and probabilities to the DataFrame
    df['BERTopic_Topics'] = topics
    df['BERTopic_Probs'] = probs.tolist()  # Convert numpy array to list
    
    # Optionally, get the topic representations (words and their importance in each topic)
    topic_info = topic_model.get_topic_info()
    
    return df, topic_model

def apply_vader_sentiment(df, text_column):
    # Initialize the VADER sentiment intensity analyzer
    analyzer = SentimentIntensityAnalyzer()
    
    # Function to analyze sentiment for a single text entry
    def analyze_sentiment(text):
        return analyzer.polarity_scores(text)
    
    # Apply sentiment analysis to the text column
    sentiment_scores = df[text_column].apply(analyze_sentiment)
    
    # Convert the sentiment scores to a DataFrame
    sentiment_df = sentiment_scores.apply(pd.Series)
    
    # Append the sentiment scores to the original DataFrame
    df = pd.concat([df, sentiment_df], axis=1)
    
    return df

def extract_named_entities_hf_split(csv_file, column_name):
    # Load the Hugging Face model and tokenizer
    model_name = "eventdata-utd/conflibert-named-entity-recognition"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    # Create a pipeline for NER with aggregation strategy
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)
    
    # Ensure the specified column exists
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the CSV file.")
    
    # Prepare lists to store the results
    entities_list = []
    entity_types_list = []

    # Function to split text into smaller chunks
    def split_text(text, max_length):
        tokens = tokenizer.tokenize(text)
        chunks = []
        for i in range(0, len(tokens), max_length):
            chunk = tokens[i:i + max_length]
            chunk_text = tokenizer.convert_tokens_to_string(chunk)
            chunks.append(chunk_text)
        return chunks

    # Iterate over the rows in the specified column with a progress bar
    for text in tqdm(df[column_name], total=len(df)):
        # Skip empty text entries
        if not isinstance(text, str) or text.strip() == "":
            entities_list.append("")
            entity_types_list.append("")
            continue
        
        # Split the text into chunks
        text_chunks = split_text(text, 510)  # 510 to leave room for special tokens

        all_entities = []
        all_entity_types = []

        # Process each chunk separately
        for chunk in text_chunks:
            ner_results = ner_pipeline(chunk)
            entities = [entity['word'] for entity in ner_results]
            entity_types = [entity['entity_group'] for entity in ner_results]

            all_entities.extend(entities)
            all_entity_types.extend(entity_types)

        entities_list.append(", ".join(all_entities))
        entity_types_list.append(", ".join(all_entity_types))

    # Add the extracted entities and their types to the DataFrame
    df['entities'] = entities_list
    df['entity_types'] = entity_types_list
    
    return df

def extract_named_entities_hf(csv_file, column_name):
    # Load the Hugging Face model and tokenizer
    model_name = "eventdata-utd/conflibert-named-entity-recognition"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    # Create a pipeline for NER with aggregation strategy
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)
    
    # Ensure the specified column exists
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the CSV file.")
    
    # Prepare lists to store the results
    entities_list = []
    entity_types_list = []

    # Iterate over the rows in the specified column with a progress bar
    for text in tqdm(df[column_name], total=len(df)):
        # Skip empty text entries
        if not isinstance(text, str) or text.strip() == "":
            entities_list.append("")
            entity_types_list.append("")
            continue
        
        ner_results = ner_pipeline(text)
        entities = [entity['word'] for entity in ner_results]
        entity_types = [entity['entity_group'] for entity in ner_results]

        entities_list.append(", ".join(entities))
        entity_types_list.append(", ".join(entity_types))

    # Add the extracted entities and their types to the DataFrame
    df['entities'] = entities_list
    df['entity_types'] = entity_types_list
    
    return df

# Example usage:
# updated_df = extract_named_entities_hf('path_to_your_file.csv', 'your_column_name')
# updated_df.to_csv('output_file.csv', index=False)
# Define the list of stopwords


def plot_top_entities(file_path, entity_types,top_n=15):
    stopwords = [ 'i', 'im', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
    'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
    'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
    'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
    'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
    'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should',
    'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn',
    'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn',
    'm', 'dms', 'nytimes', 'bbc', 'apnews', 'outlookindia','two','10','000','##']
    
    # Load the CSV file
    
    data = pd.read_csv(file_path)

    # Extract the relevant columns for organizations and persons
    entities_df = data[['entities', 'entity_types']].dropna()

    # Split the 'entities' and 'entity_types' columns into lists
    entities_df['entities'] = entities_df['entities'].apply(lambda x: x.split(', '))
    entities_df['entity_types'] = entities_df['entity_types'].apply(lambda x: x.split(', '))

    # Correct the mismatch in the counts of split columns by ensuring both columns are aligned properly
    entities_df = entities_df[entities_df['entities'].apply(len) == entities_df['entity_types'].apply(len)]

    # Explode the lists to separate rows
    entities_df = entities_df.explode(['entities', 'entity_types'])

    # Filter out stopwords and select specified entity types
    filtered_entities = entities_df[
        ~entities_df['entities'].str.lower().isin(stopwords) &
        entities_df['entity_types'].isin(entity_types)
    ]

    # Count the occurrences of each entity
    filtered_entity_counts = filtered_entities['entities'].value_counts().head(top_n)

    # Plot using Plotly
    fig = px.bar(filtered_entity_counts, 
                 x=filtered_entity_counts.index, 
                 y=filtered_entity_counts.values, 
                 labels={'x': 'Entities', 'y': 'Count'},
                 title=f'Top {top_n} {", ".join(entity_types)} (Filtered)')
    fig.update_layout(xaxis_title='Entities', yaxis_title='Count', xaxis_tickangle=-45)
    fig.show()


# Example usage:
# plot_top_entities('output_file3.csv', ['Organisation', 'Person','Weapon','Location'], stopwords, top_n=50)

def get_top_entities_df(file_path, entity_types, top_n=15):
    # Define the list of stopwords
    stopwords = [
        'i', 'im', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
        'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
        'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
        'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
        'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
        'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
        'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
        'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
        'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should',
        'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn',
        'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn',
        'm', 'dms', 'nytimes', 'bbc', 'apnews', 'outlookindia','two','10','000','##'
    ]
    # Load the CSV file
    data = pd.read_csv(file_path)

    # Extract the relevant columns for organizations and persons
    entities_df = data[['entities', 'entity_types']].dropna()

    # Split the 'entities' and 'entity_types' columns into lists
    entities_df['entities'] = entities_df['entities'].apply(lambda x: x.split(', '))
    entities_df['entity_types'] = entities_df['entity_types'].apply(lambda x: x.split(', '))

    # Correct the mismatch in the counts of split columns by ensuring both columns are aligned properly
    entities_df = entities_df[entities_df['entities'].apply(len) == entities_df['entity_types'].apply(len)]

    # Explode the lists to separate rows
    entities_df = entities_df.explode(['entities', 'entity_types'])

    # Filter out stopwords and select specified entity types
    filtered_entities = entities_df[
        ~entities_df['entities'].str.lower().isin(stopwords) &
        entities_df['entity_types'].isin(entity_types)
    ]

    # Count the occurrences of each entity
    filtered_entity_counts = filtered_entities['entities'].value_counts().head(top_n)

    # Convert the results to a DataFrame
    result_df = pd.DataFrame({
        'Entity': filtered_entity_counts.index,
        'Count': filtered_entity_counts.values
    })

    return result_df

def filter_and_plot(data, regex=None, column=None, aggregation='daily'):
    # Ensure the timestamp column is in datetime format
    data['timestamp'] = pd.to_datetime(data['timestamp'], utc=True, errors='coerce')
    
    # Filter the dataframe based on the regex if provided
    if regex is not None and column is not None:
        if column not in data.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataframe.")
        filtered_data = data[data[column].astype(str).str.contains(regex, regex=True, na=False)]
    else:
        filtered_data = data
    
    # Remove rows with NaT in the timestamp column
    filtered_data = filtered_data.dropna(subset=['timestamp'])
    
    # Define the date format based on the aggregation level
    if aggregation == 'daily':
        filtered_data['date'] = filtered_data['timestamp'].dt.date
    elif aggregation == 'weekly':
        filtered_data['date'] = filtered_data['timestamp'].dt.to_period('W').apply(lambda r: r.start_time)
    elif aggregation == 'monthly':
        filtered_data['date'] = filtered_data['timestamp'].dt.to_period('M').apply(lambda r: r.start_time)
    elif aggregation == 'yearly':
        filtered_data['date'] = filtered_data['timestamp'].dt.to_period('Y').apply(lambda r: r.start_time)
    else:
        raise ValueError("Invalid aggregation level. Choose from 'daily', 'weekly', 'monthly', or 'yearly'.")
    
    # Group by date and count the number of records per period
    daily_counts = filtered_data.groupby('date').size().reset_index(name='count')
    
    # Create a bar plot using plotly
    fig = px.bar(daily_counts, x='date', y='count', title=f"Count of Records ({aggregation.capitalize()})",
                 labels={'date': 'Date', 'count': 'Count of Records'})
    fig.update_layout(xaxis_title='Date', yaxis_title='Count of Records', title_x=0.5)
    fig.show()

# Example usage:

# Example usage:
# df = get_top_entities_df('/path/to/your/file.csv', ['Organisation', 'Person'], stopwords, top_n=15)
# print(df)
# Count noun phrases 

def is_valid_phrase(phrase):
    custom_stopwords = [
        'i', 'im', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
        'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
        'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
        'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
        'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
        'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
        'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
        'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
        'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should',
        'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn',
        'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn',
        'm', 'dms', 'nytimes', 'bbc', 'apnews', 'outlookindia','two','10','000','##'
    ]
    # Load the stopwords from NLTK
    nltk_stopwords = set(stopwords.words('english'))
    # Combine NLTK stopwords with custom stopwords
    stop_words = nltk_stopwords.union(set(custom_stopwords))
    # Split the phrase into words
    words = phrase.split()
    # Check if the phrase contains any non-stopword words
    return any(word.lower() not in stop_words for word in words)

def count_noun_phrases(file_path, column_name):
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Extract noun phrases from the specified column
    noun_phrases = df[column_name].dropna().str.split(', ')

    # Flatten the list of noun phrases
    all_noun_phrases = [phrase for sublist in noun_phrases for phrase in sublist]

    # Filter out phrases that are empty or contain only stopwords
    valid_noun_phrases = [phrase for phrase in all_noun_phrases if is_valid_phrase(phrase)]

    # Count the frequency of each valid noun phrase
    noun_phrase_counts = Counter(valid_noun_phrases)

    # Convert to a DataFrame for better visualization
    noun_phrase_counts_df = pd.DataFrame(noun_phrase_counts.items(), columns=['Noun Phrase', 'Frequency'])
    noun_phrase_counts_df = noun_phrase_counts_df.sort_values(by='Frequency', ascending=False).reset_index(drop=True)

    return noun_phrase_counts_df

# Example usage:
# file_path = 'output_file_ncs3.csv'
# column_name = 'noun_chunks'
# noun_phrase_counts_df = count_noun_phrases(file_path, column_name)

