from message_sender import MessageSender
import settings
from dnd_bot import DiscordBot


def run():
    selection = input("Would you like to run the discord bot or the OpenAI chat?\nFor the discord bot ype 'discord'\nFor OpenAI type 'open'\n")
    if selection.lower() == "discord" | "d":
        runDiscordBot()
    elif selection.lower() == "open" | "o":
        runOpenAi()
    else:
        print("I did not understand your command terminating the program.")


def runDiscordBot():
    bot = DiscordBot()
    bot.run()
def runOpenAi():
    sender = MessageSender()

    while True:
        # Message from the user
        # TODO: needs to be updated to an array of objects received from discord
        message = input()

        if message.lower() == "quit":
            break

        else:
            response = sender.send(message)
            print("DnD Oracle: ", response.content)

if __name__ == '__main__':
    run()


