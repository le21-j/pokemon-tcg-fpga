"""Small checks for legal candidates, draw labels, and partial training batches."""

import torch

from .model import MyModel, SparseVector, candidate_actions, configure_vocabulary
from .train import Sample, label_game, optimize
from ..arena import validate_action


def main():
    for minimum, maximum, size in ((0, 0, 0), (0, 3, 3), (1, 1, 7), (2, 3, 4)):
        selection = {"minCount": minimum, "maxCount": maximum, "option": [{}] * size}
        actions = candidate_actions(selection, 64)
        assert actions and len({tuple(a) for a in actions}) == len(actions)
        for action in actions:
            validate_action({"select": selection}, action)
        assert {len(a) for a in actions} == set(range(minimum, maximum + 1))
    samples = [[Sample(0, [0], None, None)], [Sample(0, [0], None, None)]]
    assert all(sample.value == 0 for sample in label_game(samples, 2))

    torch.set_num_threads(1)
    torch.manual_seed(7)
    configure_vocabulary(cards=20, attacks=10)
    model = MyModel(8, 2, 16, 1, 1)
    encoder = SparseVector()
    for _ in range(24):
        encoder.word_start()
        encoder.add(1, 1)
    decoder = SparseVector()
    decoder.word_start()
    decoder.add(1, 1)
    decoder.word_start()
    decoder.add(2, 1)
    data = [Sample(1, [0.3, -0.3], encoder, decoder)]
    before = model.encoder_fc.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    report = optimize(model, data, optimizer, batch_size=128)
    assert report[0]["updates"] == 1  # A tiny run must still optimize its partial batch.
    assert not torch.equal(before, model.encoder_fc.weight)
    print("PASS: legal cardinalities/orders, empty selections, draw targets, partial batches and changed model weights")


if __name__ == "__main__":
    main()
