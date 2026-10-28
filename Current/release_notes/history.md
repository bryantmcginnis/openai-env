# 📜 The dndbot changelog, from the beginning

Welcome to **#dndbot-release-notes**. Every time the bot gets an update, it posts what's new here: new features, how to use them, and which bugs got squashed. Here's how it got to where it is today.

## 🥚 Level 1: The Hello World Poet (Nov 2023)
The bot's first form was a chatbot on OpenAI's GPT-3.5 with one personality setting: *"a poetic assistant, skilled in only saying hello world in the various ways you can say it."* It could also add two numbers together. It had no memory and no ears, and it was not much use at a D&D table.

## 🧙 Level 5: The Great Migration (Jul 2026)
After a long rest of about two and a half years, it was rebuilt from the ground up:
- **New brain:** it switched from GPT-3.5 to **Claude**. It now remembers the conversation in each channel, so follow-up questions work. `!forget` wipes its memory of that channel.
- **New ears:** `!record` makes it join your voice channel and **transcribe everyone, live**, into #transcripts. `!stoprecording` ends it.
- *Nerd corner:* the speech-to-text runs **on the host PC's graphics card** using *Whisper large-v3*, so no audio leaves the machine for transcription. It waits for each speaker to pause before transcribing, instead of chopping audio every few seconds, which is what used to cut words in half. It also gets a cheat sheet of D&D words so "Eldritch Blast" doesn't come out as "elder cheese blast".
- *Deep nerd corner:* Discord encrypts voice end to end (a protocol called DAVE), and no released version of the bot library could hear encrypted voice. The bot runs on a not-yet-released fix from the library's developers. Getting the graphics card working on Windows meant hunting down four separate NVIDIA files, all of which failed with the same misleading error.

## 📚 Level 8: It learns our campaign (Aug 2026)
- **It knows who and what we are:** a campaign notes file teaches it our characters, NPCs, places, and the ways it tends to mishear them, so "Llyr" (pronounced *clear*) stops turning into random words.
- **Ghost-line filter:** Whisper learned from YouTube videos, so when it hears silence it sometimes invents phrases like *"Thanks for watching!"* The bot now deletes these ghost lines. Each time we spot a new one, it goes on the list.
- **`!recap`** writes a **session recap** of everything since the last one, and one gets posted automatically when recording stops. It comes in styles: `!recap sarcastic`, `epic`, `noir`, `funny`, `short`, `long`, `factual`.
- **Personal recaps:** `!recap Sykth` DMs you your character's night: your decisions, what you learned, and what's still hanging over you. `!recap @Justin Llyr` sends one to someone else.
- **Proper dice:** `!roll 1d20+5`, `!roll 4d6kh3` (stat rolling), `!roll 1d20 adv`. Dropped dice are shown ~~struck through~~.
- **Tidier transcripts:** lines from the same person are merged into one message instead of a wall of tiny ones.

## 🛡️ Level 9: Constitution save (Sep 2026)
- **It survives wifi drops:** a recap that took minutes used to die completely if the connection blinked partway through. Now it waits, retries, and carries on from where it was.
- **DM the bot:** commands like `!recap` work in a private message too, so you can get recaps without cluttering the channel.
- **`!shutdown`** turns the bot off cleanly. Only the bot's owner can use it.

## 🩺 Level 11: The bot gets a pulse check (21 Oct 2026)
- **Voice health monitor:** sometimes Discord's encryption gets out of sync and the bot goes deaf without noticing. That used to cost us whole sessions of transcript. Now the bot checks its own hearing. If the audio stops making sense it first re-syncs quietly, then tries **one** rejoin of the voice channel, and only then posts a single ⚠️ in chat. It never spams.
- **Better logs:** a separate "black box" log keeps just the important events, so when something breaks we can see why, and big error floods are condensed so they can't fill the disk.
- **Brain upgrade:** chat runs on the fast **Claude Haiku 5.5** and recaps on the heavyweight **Claude Opus 5.5**.

## 🧠 Level 13: Long-term memory (24 Oct 2026)
- **Ask about the session:** `@dndbot what did Sykth roll when he attacked?` answers from what was actually said tonight, with timestamps, plus older recaps. If it wasn't said, it says *"I didn't catch that"* instead of making something up.
- **Open threads board** in #potjf-to-do: after each recap the bot updates our list of unresolved hooks, mysteries and promises, and ticks off solved ones with how they were solved (and a quip). Commands: `!threads`, `!threads add`, `!threads resolve 3 how`, `!threads reopen 3`, `!threads scan`.
- **Quote board** in #quotes: react ⭐ to a transcript line, type `!quote` (or `!quote Thia`), or let the recap pick the best lines. *Nerd note:* every recap quote is checked word for word against the transcript, so it can't put words in anyone's mouth.
- **Character names, not player names:** the transcript gets relabelled before recaps see it (`wildgoose` becomes **Thia (Goose)**), so recaps use the right names and pronouns. The best out-of-character moments go in a new **IRL** section.
- **`!correct`**: DM the bot what a recap got wrong and it **edits the original recap in place**, plus any affected threads or quotes. It also remembers the correction, so the mistake can't come back later.
- **Phoenix mode:** if the bot crashes it restarts itself, and if it was recording it **rejoins voice and carries on**, then tells you how big the gap in the transcript is.
- *Nerd note:* it also checks the PC's clock against Discord's at startup, after a clock running two minutes fast confused a debugging session.

## 🔮 Coming up
Each update to the bot now gets its own post here, written automatically from the code changes.
