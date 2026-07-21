import os

# Terms that generic speech models routinely mangle because they never show
# up in everyday training data. Feeding these to Whisper as an
# `initial_prompt` biases the decoder toward this vocabulary/spelling when
# it hears something ambiguous -- it does not need to match what's spoken
# verbatim, it just needs to nudge word choice in the right direction.
_CORE_TERMS = [
    # Ability scores & mechanics
    "Strength", "Dexterity", "Constitution", "Intelligence", "Wisdom", "Charisma",
    "armor class", "AC", "saving throw", "advantage", "disadvantage",
    "proficiency bonus", "hit points", "temporary hit points", "death save",
    "short rest", "long rest", "initiative", "opportunity attack",
    "concentration", "spell slot", "cantrip", "ritual",
    # Classes
    "Barbarian", "Bard", "Cleric", "Druid", "Fighter", "Monk", "Paladin",
    "Ranger", "Rogue", "Sorcerer", "Warlock", "Wizard", "Artificer",
    # Races
    "Dragonborn", "Dwarf", "Elf", "Gnome", "Half-Elf", "Halfling",
    "Half-Orc", "Human", "Tiefling", "Aasimar", "Goliath", "Tabaxi", "Firbolg",
    # Common spells (frequently mangled by generic ASR)
    "Fireball", "Eldritch Blast", "Counterspell", "Misty Step", "Cure Wounds",
    "Guidance", "Shield", "Magic Missile", "Hold Person", "Sanctuary",
    "Bless", "Bane", "Polymorph", "Wild Shape", "Sneak Attack",
    "Divine Smite", "Rage", "Second Wind", "Action Surge", "Bardic Inspiration",
    # Common monsters
    "goblin", "kobold", "beholder", "mind flayer", "owlbear", "displacer beast",
    "githyanki", "tarrasque", "lich", "vampire", "werewolf", "dragon wyrmling",
    # Dice / table jargon
    "d20", "d12", "d10", "d8", "d6", "d4", "nat twenty", "natural one",
    "DM", "Dungeon Master", "party", "campaign", "homebrew", "loot",
]


def build_prompt(extra_names: str | None = None) -> str:
    """Build the Whisper `initial_prompt` for a D&D session.

    `extra_names` is a comma-separated list layered on top of the core 5e
    glossary above -- put player character names, recurring NPCs, and
    homebrew place names here since those are exactly what a generic model
    has no chance of getting right. Falls back to the CAMPAIGN_VOCAB env
    var so it doesn't need to be passed in explicitly every time.
    """
    terms = list(_CORE_TERMS)

    for source in (extra_names, os.getenv("CAMPAIGN_VOCAB")):
        if source:
            terms.extend(name.strip() for name in source.split(",") if name.strip())

    return "Dungeons and Dragons 5th edition game session. " + ", ".join(terms) + "."
