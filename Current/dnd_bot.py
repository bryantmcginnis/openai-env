# This example requires the 'members' and 'message_content' privileged intents to function.

import discord
import random
import settings

from discord.ext import commands
from message_sender import MessageSender
from response_splitter import ResponseSplitter

class DiscordBot:

    def __init__(self):

        self.description = '''An example bot to showcase the discord.ext.commands extension
        module.
        There are a number of utility commands being showcased here.'''

        self.intents = discord.Intents.default()
        self.intents.members = True
        self.intents.message_content = True

        self.bot = commands.Bot(command_prefix='?', description=self.description, intents=self.intents)
        self.sender = MessageSender()
        self.splitter = ResponseSplitter()

        self.register_events()
        self.register_commands()

    def run(self):
        self.bot.run(settings.DISCORD_API_SECRET)

    def register_events(self):
        @self.bot.event
        async def on_ready():
            print(f'Logged in as {self.bot.user} (ID: {self.bot.user.id})')
            print('------')

    def register_commands(self):

        @self.bot.command()
        async def chat(ctx, *, message: str):
            #Send the user a message letting them know the response is being generated
            thinking_message = self.sender.send("come up with short a dnd related loading message like from a loading screen in a game")
            await ctx.send(thinking_message.content)

            #Generate the response from openAI
            response = self.sender.send(message)
            #Split the responses into small chunks manageable by discord's response length limit and return them.
            split_responses = self.splitter.split(response.content)
            for chunk in split_responses:
                await ctx.send(chunk)

        @self.bot.command()
        async def add(ctx, left: int, right: int):
            """Adds two numbers together."""
            await ctx.send(left + right)


        @self.bot.command()
        async def roll(ctx, dice: str):
            """Rolls a dice in NdN format."""
            try:
                rolls, limit = map(int, dice.split('d'))
            except Exception:
                await ctx.send('Format has to be in NdN!')
                return

            result = ', '.join(str(random.randint(1, limit)) for r in range(rolls))
            await ctx.send(result)


        @self.bot.command(description='For when you wanna settle the score some other way')
        async def choose(ctx, *choices: str):
            """Chooses between multiple choices."""
            await ctx.send(random.choice(choices))


        @self.bot.command()
        async def repeat(ctx, times: int, content='repeating...'):
            """Repeats a message multiple times."""
            for i in range(times):
                await ctx.send(content)


        @self.bot.command()
        async def joined(ctx, member: discord.Member):
            """Says when a member joined."""
            await ctx.send(f'{member.name} joined {discord.utils.format_dt(member.joined_at)}')


        # @self.bot.group()
        # async def cool(ctx):
        #     """Says if a user is cool.
        #
        #     In reality this just checks if a subcommand is being invoked.
        #     """
        #     if ctx.invoked_subcommand is None:
        #         await ctx.send(f'No, {ctx.subcommand_passed} is not cool')


        # @self.cool.command(name='bot')
        # async def _bot(ctx):
        #     """Is the bot cool?"""
        #     await ctx.send('Yes, the bot is cool.')