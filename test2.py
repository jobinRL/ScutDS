from skmultiflow.data import WaveformGenerator
from skmultiflow.trees import HoeffdingTree
from skmultiflow.evaluation import EvaluatePrequential

import scutds
# 1. Create a stream
stream = WaveformGenerator()
stream.prepare_for_use()

# 2. Instantiate the HoeffdingTree classifier
sd = scutds.ScutDS()

# 3. Setup the evaluator
evaluator = EvaluatePrequential(show_plot=False,
                                pretrain_size=1000,
                                batch_size = 1000,
                                max_samples=10000)

# 4. Run evaluation
evaluator.evaluate(stream=stream, model=sd)
