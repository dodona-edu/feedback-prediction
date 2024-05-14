# If True, AST's are used. If False concrete parse trees are used.
USE_AST = True

# The minimum support value that TreeMiner should use to determine if an embedded subtree is a pattern.
TREEMINER_MIN_SUPPORT = 0.7

# If True, identifying nodes are used on top of patterns to improve predictions.
USE_IDENTIFYING_NODES = False

# The minimum amount of subtrees needed before starting to search for TreeMiner patterns.
MIN_PATTERN_SUBTREES = 3

# The maximum amount of subtrees a node is allowed to be in to be considered an identifying node.
MAX_IDENTIFYING_SUBTREES = 3

# The random seed used to create a train-test split, or None for no seed.
RANDOM_SEED = 314159
