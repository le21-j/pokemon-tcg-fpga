"""Run `python check_agent.py` before sharing your inference folder."""

from agent import agent, start_game


def main():
    start_game([3] * 60)  # Interface fixture, not a valid playable Pokémon deck.
    for types, minimum, maximum in (([14, 13, 8], 1, 1), ([1, 2, 3], 0, 2), ([], 0, 0)):
        observation = {"select": {"option": [{"type": t} for t in types],
                                   "minCount": minimum, "maxCount": maximum}}
        action = agent(observation)
        assert isinstance(action, list)
        assert all(type(i) is int for i in action)
        assert minimum <= len(action) <= maximum
        assert len(action) == len(set(action))
        assert all(0 <= i < len(types) for i in action)
    print("PASS: Python integer indices, variable option counts, optional and empty selections")


if __name__ == "__main__":
    main()
