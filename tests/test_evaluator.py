from deep_lst.evaluation.evaluator import Evaluator


class _DummyEvaluator(Evaluator):
    def _compute(self):
        pass
