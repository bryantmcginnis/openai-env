from message_sender import MessageSender

if __name__ == '__main__':
     
    # Message from the user
    # TODO: needs to be updated to an array of objects received from discord
    chatting = True
    sender = MessageSender()
    
    while True:
        message = input()

        if message.lower() == "quit":
            break

        else:
            response = sender.send(message)
            # print("DnD Oracle: ", response.strip("\n").strip())
            print("DnD Oracle: ", response.content)