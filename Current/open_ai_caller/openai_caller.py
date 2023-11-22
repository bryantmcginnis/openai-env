from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class OpenAiChatCaller:
  client = OpenAI()

  def call(self, message: str):

    response = self.client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": "You are a games night assistant with expertise in DnD 5e, you are a rules master, monster codex, and note taker extraordinaire."},
        {"role": "user", "content": message}
      ]
    )

    return response.choices[0].message