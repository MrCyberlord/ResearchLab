"""Safe failure vocabulary for versioned evaluation summaries."""


class EvaluationArtifactError(Exception):
    """Base class whose details are never returned to API clients."""


class EvaluationArtifactMissing(EvaluationArtifactError):
    pass


class EvaluationArtifactInvalid(EvaluationArtifactError):
    pass


class EvaluationArtifactAccessDenied(EvaluationArtifactError):
    pass


class EvaluationArtifactIncomplete(EvaluationArtifactError):
    pass


class EvaluationArtifactIncompatible(EvaluationArtifactError):
    pass
