from openai_caller import OpenAiChatCaller
#comment for git

class MessageSender:

    # Maybe needs to be a singleton?
    textCaller = OpenAiChatCaller()

    # Message from the user
    # TODO: needs to be updated to an array of objects received from discord

    # TODO: determine input type & call appropriate OpenAI Assistant
    # MVP version just handing str input 
    def send(self, message: str):
        return self.textCaller.call(message)
