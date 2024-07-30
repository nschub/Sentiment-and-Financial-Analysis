import logging
import time
from openai import OpenAI
from utils import save_response_to_file, get_api_key

class SentimentAnalysisSession:
    def __init__(self, gpt_name, model, instructions):
        self.gpt_name = gpt_name
        self.model = model
        self.api_key = get_api_key()
        self.instructions = instructions
        self.client = OpenAI(api_key=self.api_key)
        self.tokens_this_minute = 0
        self.requests_this_minute = 0

    def analyze_sentiment(self, text, prompt, log_file='response_log.txt'):
        tokens_in_text = len(text)
        if self.tokens_this_minute + tokens_in_text > 400000:
            time_to_next_minute = 60 - time.time() % 60
            time.sleep(time_to_next_minute)
            self.tokens_this_minute = 0

        if self.requests_this_minute >= 500:
            time_to_next_minute = 60 - time.time() % 60
            time.sleep(time_to_next_minute)
            self.requests_this_minute = 0


        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.instructions},
                    {"role": "user", "content": text},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=60,
                top_p=1,
                temperature=0

            )

            save_response_to_file(response.model_dump_json(indent=2), log_file)
            sentiment_score = response.choices[0].message.content

            self.tokens_this_minute += tokens_in_text
            self.requests_this_minute += 1

            return sentiment_score
        except Exception as e:
            logging.error(f"Error in analyzing sentiment: {e}")
            return "Error"

    def close_session(self):
        logging.info("Session closed.")

