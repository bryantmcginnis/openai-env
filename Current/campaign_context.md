<!--
================================================================================
CAMPAIGN CONTEXT

Free-form notes handed to the recap writer (Claude) before it reads the
transcript. No format rules, no parsing -- write plain prose, it gets sent
as-is. HTML comments like this one are stripped, so these instructions
aren't included.

THIS IS NOT THE VOCAB FILE. Two different things:

  campaign_vocab.txt   -> the local Whisper model. Hard 223-token budget.
                          Names only. Affects how words are HEARD.
  campaign_context.md  -> Claude, when writing the recap. Room for a page or
                          two. Prose. Affects how the session is UNDERSTOOD.

So put "who the DM is" here, not there. There's no meaningful token pressure
(a page costs about a penny per session), but don't pad it either -- write
what actually helps someone retell the night accurately.

Takes effect on the next bot restart.
================================================================================
-->

# Who's at the table

<!--
The single most valuable thing in this file. The transcript labels every line
with the speaker's DISCORD DISPLAY NAME, so without this the recap sees
"gossip repeater said..." and has no idea whether that's the DM narrating or
a player in character.

Your Discord names, as seen in this session's transcript, are below. Fill in
who they are and which character they play. Add the rest of the table.
-->

- **gossip repeater** — Joe, the DM, Voices most NPC's that aren't keeping or directing party NPC's, narrates and sets the scenes, asks the party what we're up to and what we should do next, sometime swaps between in game and out of game contexts, calls for rolls. When this player describes a statement its happening, if its a question it might not have happened. 
- **wildgoose** — Goose - the resident Wizard, playing Thia, a leggy elf female with massive damage spells. 
- **Justin** - Justin, playing Llyr Mabmoor, a welsh named ancient half elf druid, pronounced 'clear'
-**Sykth** — me, Peter, Our resident halfling rogue, somewhat brash and forward thanks to his swashbuckler subclass. 

<!--
Speaker labels. The bot reads these lines to relabel every transcript line
before the recap sees it, so "wildgoose: ..." becomes "Thia (Goose): ...".
Format: - Discord name = Character (player), Character: pronouns, player: pronouns
Name whose pronouns are whose: a bare "she/her" after "(Goose)" once got
read as Goose's, and an IRL bullet called Goose "her".
-->
- gossip repeater = DM (Joe), Joe: he/him
- wildgoose = Thia (Goose), Thia: she/her, Goose: he/him
- Justin = Llyr (Justin)
- Sykth = Sykth (Peter), Sykth: he/him, Peter: he/him

A character's pronouns are not their player's. Thia is a woman (she/her);
Goose, who plays her, is a man (he/him). In the story, Thia is "she". In
IRL bullets about Goose himself, Goose is "he".

Out of game: Joe, the DM, practises law in real life.

<!-- Example of the shape to aim for:
- **gossip repeater** — Peter. The DM. Voices every NPC, narrates scenes,
  and calls for rolls. When this speaker describes something happening, it
  is happening in the world, not a player suggesting it.
- **wildgoose** — Sam, playing Llyr Mabmoor, a Welsh-named druid. Llyr is
  pronounced "clear", which the transcriber mangles constantly.
-->


# The characters

<!--
A line each: name, race/class, and the one trait that shapes how they act.
Enough that the recap can tell whose moment it was and why it mattered.
-->

- **Sykth** — Halfling / Rogue - Swashbuckler, face of the party... sometimes. likes to kill. 
- **Thia** — Elf / Wizard - huge and immense focus on magic and magical items, has a gem fused in her eyesocket, a wand fused to her hand. she seems a bit of a slave to the trade. 
- **Llyr Mabmoor** — Half Elf / Druid - an old deft hand, worships two opposing gods in an attempt to create a new philosophy called dualathism - to bring balance to the world.  

The party is **The Partnership of the Jade Frog** - known from far and wide due to the early EP of bardic songs written of our early adventures. 


# The premise

<!--
Two or three sentences. What's the campaign about, where are they, what are
they trying to do? This is what lets the recap open with a hook instead of
"the party continued their journey".
-->


# Where we are right now

<!--
The most recently useful section, and the one worth updating between
sessions. What's the current arc? Who did they just betray? What are they
walking into? Prevents the recap misreading a callback as a new event.
-->


# Recurring NPCs and factions

<!--
One line each on who matters and how they relate to the party. Named NPCs
are already in campaign_vocab.txt for the transcriber -- this is about who
they ARE.
-->

- **Nessus Thorne** —
- **Declo** —
- **Professor Emmett** —
- **Phalahan Firstnote** —
- **The Twelve Families** —


# Tone and house style

<!--
Optional. How the recap should read, if the default (dry, affectionate,
comic timing from what actually happened) isn't right for your table.
Also note anything to leave out -- in-jokes that don't travel, a player who
would rather not be quoted, running bits you're sick of.
-->
