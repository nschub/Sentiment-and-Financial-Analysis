import logging
from test_sentiment_analysis import SentimentAnalysisSession
from utils import generate_filename
from datetime import datetime
import pandas as pd
import os


def process_dataframe(df, bank_name, input_filepath):
    # Map bank names to their corresponding output folder names
    bank_folder_map = {
        'Credit Suisse': 'CS',
        'UBS': 'UBS'
    }
    output_directory = f'Output_Data/{bank_folder_map[bank_name]}'
    os.makedirs(output_directory, exist_ok=True)  # Ensure the output directory exists

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Construct the output filename with timestamp
    base_filename = os.path.basename(input_filepath)  # Gets the filename from the full path
    output_filename = base_filename.replace('.csv', f'_{timestamp}.csv')
    output_filepath = os.path.join(output_directory, output_filename)

    session = SentimentAnalysisSession(
        gpt_name="GPT-Sentiment-Analyzer",
        model="gpt-4-turbo-2024-04-09",
        instructions="Forget all previous instructions. Pretend you are a sentiment analysis expert."
    )
    log_file = generate_filename(bank_folder_map[bank_name], 'txt', 'Output_Data/Logfiles', timestamp)

    prompt = f"Analyze the sentiment of this article towards {bank_name}. Score the sentiment as either 1 for positive, -1 for negative, or 0 for neutral. Provide the sentiment score, followed by a brief explanation in less than 20 words, separating them with a period. Example: '[insert sentiment score]. [insert explanation]'."

    output_df = df.copy()

    logging.info(f"Processing dataframe for {bank_name} from {input_filepath}")
    for index, row in df.iterrows():
        sentiment = session.analyze_sentiment(row['article_content'], prompt, log_file)
        output_df.at[index, 'Sentiment'] = sentiment

    output_df.to_csv(output_filepath, index=False)
    session.close_session()


def process_directory(directory, bank_name):
    logging.info(f"Processing directory {directory} for {bank_name}")
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            process_dataframe(pd.read_csv(filepath), bank_name, filepath)


def main():
    logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
    logging.info("Which bank's data would you like to process?")
    logging.info("1: Credit Suisse")
    logging.info("2: UBS")
    choice = input("Enter your choice (1 or 2): ")

    if choice == '1':
        process_directory('Data/CS', 'Credit Suisse')
    elif choice == '2':
        process_directory('Data/UBS', 'UBS')
    else:
        logging.error("Invalid choice. Please run the program again and select either 1 or 2.")


if __name__ == '__main__':
    main()
