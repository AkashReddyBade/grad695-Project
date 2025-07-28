import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
import requests
from transformers import pipeline
from openai import OpenAI
from prettytable import PrettyTable

# Parameters
vocab_size = 10000  # Vocabulary size
maxlen = 300        # Maximum length of review
embedding_size = 50 # Dimension of the embedding vector

# Load the dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# Pad sequences to ensure uniform input size
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# Build the model
model = Sequential()
model.add(Embedding(vocab_size, embedding_size, input_length=maxlen))
model.add(GlobalAveragePooling1D())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# Train the model
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=512,
                    validation_data=(x_test, y_test),
                    verbose=2)

# Evaluate the model
results = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {results[1] * 100:.2f}%")

# Initialize the sentiment analysis model
classifier = pipeline("sentiment-analysis")

def get_places(query, api_key):
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        'query': f"hotels in {query}",
        'fields': 'formatted_address,name,place_id,rating,user_ratings_total',
        'key': api_key
    }
    response = requests.get(url, params=params)
    results = response.json().get('results', [])
    return results[:10]  # Get top 10 hotels

def get_reviews(place_id, api_key):
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        'place_id': place_id,
        'fields': 'review',
        'key': api_key
    }
    response = requests.get(url, params=params)
    reviews = response.json().get('result', {}).get('reviews', [])
    return reviews

def analyze_reviews(reviews):
    sentiments = []
    for review in reviews:
        result = classifier(review['text'][:512])  # Truncate to 512 tokens if needed
        sentiments.append(result[0])
    return sentiments

def calculate_sentiment_stats(sentiments):
    positive_count = sum(1 for s in sentiments if s['label'] == 'POSITIVE')
    negative_count = sum(1 for s in sentiments if s['label'] == 'NEGATIVE')
    total = len(sentiments)

    if total == 0:
        return 0, 0, 0

    positive_percent = (positive_count / total) * 100
    negative_percent = (negative_count / total) * 100
    avg_confidence = sum(s['score'] for s in sentiments) / total

    return positive_percent, negative_percent, avg_confidence

def extract_common_complaints(reviews, sentiments):
    negative_reviews = [review['text'] for review, sentiment in zip(reviews, sentiments) if sentiment['label'] == 'NEGATIVE']

    # Simple keyword-based approach (could be enhanced with NLP)
    common_words = {
        'clean': ['dirty', 'clean', 'unclean', 'hygiene', 'filthy'],
        'service': ['service', 'staff', 'rude', 'unhelpful', 'friendly'],
        'noise': ['noise', 'loud', 'quiet', 'disturbance'],
        'food': ['food', 'breakfast', 'dinner', 'restaurant', 'meal'],
        'comfort': ['bed', 'comfortable', 'uncomfortable', 'pillow', 'mattress']
    }

    complaints = {}
    for category, keywords in common_words.items():
        count = sum(1 for review in negative_reviews if any(keyword in review.lower() for keyword in keywords))
        if count > 0:
            complaints[category] = count

    # Get top 3 complaints
    sorted_complaints = sorted(complaints.items(), key=lambda x: x[1], reverse=True)[:3]
    return [item[0] for item in sorted_complaints]

client = OpenAI(api_key='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')

def generate_analysis_summary(hotel_data):
    prompt = """Based on the following hotel review sentiment analysis, write a concise one-paragraph
    summary about hotels in this area. Focus specifically on key points for improvement that multiple
    hotels share. Organize your response with clear, actionable insights. Here's the data:\n\n"""

    for hotel in hotel_data:
        prompt += f"Hotel: {hotel['name']}\n"
        prompt += f"Positive reviews: {hotel['positive_percent']:.1f}%\n"
        prompt += f"Negative reviews: {hotel['negative_percent']:.1f}%\n"
        if hotel.get('common_complaints'):
            prompt += f"Common complaints: {', '.join(hotel['common_complaints'])}\n"
        prompt += "\n"

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a hospitality analyst providing concise, actionable insights."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating summary: {e}")
        return "Could not generate summary due to an error."

def main():
    google_api_key = 'AIzaSyCjmVdtwPi7Ez5CKRL_o5NvL673Kzmh5cs'
    location = input("Enter a location to search for hotels: ")

    # Get top 10 hotels in the location
    hotels = get_places(location, google_api_key)

    if not hotels:
        print("No hotels found in this location.")
        return

    hotel_data = []
    overall_table = PrettyTable()
    overall_table.field_names = ["Hotel", "Positive %", "Negative %", "Avg Confidence", "Common Complaints"]

    # Analyze each hotel
    for hotel in hotels:
        print(f"\nAnalyzing reviews for: {hotel['name']}...")
        reviews = get_reviews(hotel['place_id'], google_api_key)

        if not reviews:
            print(f"No reviews found for {hotel['name']}")
            continue

        sentiments = analyze_reviews(reviews)
        positive_percent, negative_percent, avg_confidence = calculate_sentiment_stats(sentiments)
        common_complaints = extract_common_complaints(reviews, sentiments)

        hotel_info = {
            'name': hotel['name'],
            'positive_percent': positive_percent,
            'negative_percent': negative_percent,
            'avg_confidence': avg_confidence,
            'common_complaints': common_complaints
        }
        hotel_data.append(hotel_info)

        overall_table.add_row([
            hotel['name'],
            f"{positive_percent:.1f}%",
            f"{negative_percent:.1f}%",
            f"{avg_confidence:.2f}",
            ", ".join(common_complaints) if common_complaints else "N/A"
        ])

    print("\nOverall Analysis of Hotels in", location)
    print(overall_table)

    # Generate summary using OpenAI
    print("\nGenerating summary analysis...")
    summary = generate_analysis_summary(hotel_data)
    print("\n=== Area Hotel Improvement Summary ===")
    print(summary)

if __name__ == "__main__":
    main()