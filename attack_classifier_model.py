# ============================================================
# STAGE 2 - ATTACK CLASSIFICATION MODEL
#
# A classical (non-quantum) multi-class classifier on the same 11 flow
# features used by the Hybrid QNN. It only ever runs on flows that Stage 1
# (the Hybrid QNN) has already labeled ATTACK, so it stays a lightweight
# second opinion rather than a second full detection engine.
# ============================================================

import torch.nn as nn


class AttackClassifier(nn.Module):
    """MLP: 11 flow features -> N attack-type classes.

    Default architecture: 128 -> 64 -> 32 hidden layers, each followed by
    BatchNorm1d, ReLU, and Dropout(0.3). app.py instantiates this with no
    `hidden`/`dropout` override, so these defaults ARE the architecture
    used for live-capture inference -- keep them in sync with whatever
    train_attack_classifier.py trains with.

    num_classes is not hardcoded to 16: it's set from however many
    canonical attack types are actually present in the training data
    (see train_attack_classifier.py), and the trained class list is
    saved alongside the model so inference always matches.

    BatchNorm1d requires app.py to call model.eval() before inference
    (it already does) so single-flow predictions use running statistics
    instead of failing on a batch size of 1.
    """

    def __init__(self, input_dim=11, num_classes=16, hidden=(128, 64, 32), dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)