"""Single source of truth for subject partition validation."""

from types import MappingProxyType

SUBJECT_SPLITS = MappingProxyType({
    "S1": "train", "S5": "train", "S6": "train", "S7": "train",
    "S8": "validation", "S9": "test", "S11": "test",
})

# Pilot subject-ID split, not an official benchmark or verified actor identity split.
CMU_SUBJECT_SPLITS = MappingProxyType({
    "CMU:02": "train", "CMU:05": "train", "CMU:06": "train",
    "CMU:13": "validation", "CMU:14": "test",
})


def subject_split(subject):
    return SUBJECT_SPLITS.get(subject, CMU_SUBJECT_SPLITS.get(subject))


def sequence_split(reference):
    for frame in reference.frames:
        for field in ("subject", "split", "dataset"):
            value = frame.metadata.get(field)
            if value is not None and not isinstance(value, str):
                raise ValueError("subject/split/dataset labels must be strings")
    subjects = {frame.metadata.get("subject") for frame in reference.frames}
    splits = {frame.metadata.get("split") for frame in reference.frames}
    datasets = {frame.metadata.get("dataset") for frame in reference.frames}
    if subjects == {None} and splits == {None} and "Human3.6M" not in datasets:
        return "unlabelled", None
    if datasets not in ({"Human3.6M"}, {"CMU"}):
        raise ValueError("labelled split validation requires a supported dataset")
    if len(subjects) != 1 or len(splits) != 1:
        raise ValueError("reference contains mixed subject/split labels")
    subject, split = next(iter(subjects)), next(iter(splits))
    policy = SUBJECT_SPLITS if datasets == {"Human3.6M"} else CMU_SUBJECT_SPLITS
    if subject not in policy or policy[subject] != split:
        raise ValueError("reference subject violates fixed dataset split")
    return split, subject
