"""Dream bucket enumeration module."""

from enum import Enum


class DreamBucketEnum(str, Enum):
    """Enumeration of digest memory buckets used by dream integration.

    ``MUSCLE`` is the ops muscle-memory bucket (SOPs promoted after repeated
    successful reuse). It is never an extract/integrate target: units only
    reach it through usage-driven promotion, so extract prompts exclude it.
    """

    PROCEDURE = "procedure"
    PERSONAL = "personal"
    WIKI = "wiki"
    MUSCLE = "muscle"
