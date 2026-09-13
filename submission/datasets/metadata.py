OBJECT_NAMES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

# Object IDs 3, 8, 10 and 17 have no part categories in Pascal-Part-116.
PARTS_BY_OBJECT = {
    0: (
        "aeroplane",
        ("body", "stern", "wing", "tail", "engine", "wheel"),
    ),
    1: (
        "bicycle",
        ("wheel", "saddle", "handlebar", "chainwheel", "headlight"),
    ),
    2: (
        "bird",
        ("wing", "tail", "head", "eye", "beak", "torso", "neck", "leg", "foot"),
    ),
    4: ("bottle", ("body", "cap")),
    5: (
        "bus",
        (
            "wheel", "headlight", "front", "side", "back",
            "roof", "mirror", "license plate", "door", "window",
        ),
    ),
    6: (
        "car",
        (
            "wheel", "headlight", "front", "side", "back",
            "roof", "mirror", "license plate", "door", "window",
        ),
    ),
    7: (
        "cat",
        ("tail", "head", "eye", "torso", "neck", "leg", "nose", "paw", "ear"),
    ),
    9: (
        "cow",
        ("tail", "head", "eye", "torso", "neck", "leg", "ear", "muzzle", "horn"),
    ),
    11: (
        "dog",
        (
            "tail", "head", "eye", "torso", "neck",
            "leg", "nose", "paw", "ear", "muzzle",
        ),
    ),
    12: (
        "horse",
        ("tail", "head", "eye", "torso", "neck", "leg", "ear", "muzzle", "hoof"),
    ),
    13: ("motorbike", ("wheel", "saddle", "handlebar", "headlight")),
    14: (
        "person",
        (
            "head", "eye", "torso", "neck", "leg", "foot", "nose",
            "ear", "eyebrow", "mouth", "hair", "lower arm",
            "upper arm", "hand",
        ),
    ),
    15: ("pottedplant", ("pot", "plant")),
    16: (
        "sheep",
        ("tail", "head", "eye", "torso", "neck", "leg", "ear", "muzzle", "horn"),
    ),
    18: (
        "train",
        ("headlight", "head", "front", "side", "back", "roof", "coach"),
    ),
    19: ("tvmonitor", ("screen",)),
}

# Object categories held out for open-vocabulary evaluation.
UNSEEN_OBJECT_NAMES = {
    "bird",
    "car",
    "dog",
    "motorbike",
    "sheep",
}

PART_CATEGORIES = []

for object_id, (object_name, part_names) in PARTS_BY_OBJECT.items():
    for part_name in part_names:
        part_id = len(PART_CATEGORIES)

        PART_CATEGORIES.append(
            {
                "part_id": part_id,
                "object_id": object_id,
                "object_name": object_name,
                "part_name": part_name,
                "full_name": f"{object_name}'s {part_name}",
                "evaluation_split": (
                    "unseen"
                    if object_name in UNSEEN_OBJECT_NAMES
                    else "seen"
                ),
            }
        )

assert len(OBJECT_NAMES) == 20
assert len(PART_CATEGORIES) == 116


def get_part_category(part_id: int) -> dict:
    if not 0 <= part_id < len(PART_CATEGORIES):
        raise ValueError(f"Invalid Pascal-Part-116 part ID: {part_id}")

    return PART_CATEGORIES[part_id]