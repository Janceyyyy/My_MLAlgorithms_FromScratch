This document outlines the primary framework for data acquisition, model implementation, and outcomes visualization. It applies the model to two distinct datasets:

1. **Spambase Dataset**: Aimed at developing a model capable of discerning spam emails from non-spam, analyzing attributes like word frequency and capital letter usage within the emails. Available at https://archive.ics.uci.edu/ml/datasets/spambase.

2. **Chess Dataset**: Incorporates data from "chess.csv," with each entry comprising 36 features that depict the chessboard's current configuration. The objective is leveraging Decision Trees to predict the feasibility of a win for the white side. Accessible at https://archive.ics.uci.edu/ml/datasets/Chess+%28King-Rook+vs.+King-Knight%29.


Results for Training Error:
(a) Average Training Loss (not-pruned): 0.0000
(b) Average Test Loss (not-pruned): 0.0351
(c) Average Training Loss (pruned): 0.0070
(d) Average Test Loss (pruned): 0.0301


Results for Entropy:
(a) Average Training Loss (not-pruned): 0.0000
(b) Average Test Loss (not-pruned): 0.0134
(c) Average Training Loss (pruned): 0.0020
(d) Average Test Loss (pruned): 0.0117


Results for Gini:
(a) Average Training Loss (not-pruned): 0.0000
(b) Average Test Loss (not-pruned): 0.0142
(c) Average Training Loss (pruned): 0.0020
(d) Average Test Loss (pruned): 0.0125


Results for Training Error:
(a) Average Training Loss (not-pruned): 0.0140
(b) Average Test Loss (not-pruned): 0.1365
(c) Average Training Loss (pruned): 0.0280
...
(c) Average Training Loss (pruned): 0.0210
(d) Average Test Loss (pruned): 0.1300