"""A small, flat summary of a poke-env battle, used only to COMPARE two views of a battle.

This is a spike helper (a throwaway experiment), not the real `PublicState` converter. It turns
a poke-env `Battle` into plain dicts so we can diff "the battle as rebuilt from a replay" against
"the battle as the live player saw it" turn by turn.

Fields are split into PUBLIC ones (anyone watching could know them) and PRIVATE ones (only the
owner knows them, e.g. your own moves before you use them). When comparing a live player's own
side, only public fields are meaningful; for the opponent side, everything poke-env tracks is
public by construction.
"""

from poke_env.battle import Battle

PUBLIC_FIELDS = ("hp", "status", "fainted", "active", "boosts", "tera_type", "terastallized")
REVEALED_FIELDS = ("moves", "item", "ability")  # public on the opponent side only


def mon_summary(mon) -> dict:
    return {
        "species": mon.species,
        "hp": round(mon.current_hp_fraction, 2),
        "hp_raw": f"{mon.current_hp}/{mon.max_hp}",
        "status": mon.status.name if mon.status else None,
        "fainted": mon.fainted,
        "active": bool(mon.active),
        "boosts": {k: v for k, v in mon.boosts.items() if v},
        "moves": sorted(mon.moves),
        "item": mon.item,
        "ability": mon.ability,
        "tera_type": mon.tera_type.name if mon.tera_type else None,
        "terastallized": mon.is_terastallized,
    }


def side_summary(team: dict) -> dict:
    """Map species -> summary. Species (not nickname) is the key both views agree on."""
    return {mon.species: mon_summary(mon) for mon in team.values()}


def battle_summary(battle: Battle) -> dict:
    return {
        "turn": battle.turn,
        "me": side_summary(battle.team),
        "opp": side_summary(battle.opponent_team),
        "my_side_conditions": sorted(c.name for c in battle.side_conditions),
        "opp_side_conditions": sorted(c.name for c in battle.opponent_side_conditions),
        "weather": sorted(w.name for w in battle.weather),
        "fields": sorted(f.name for f in battle.fields),
        "opp_used_tera": battle.opponent_used_tera,
    }
