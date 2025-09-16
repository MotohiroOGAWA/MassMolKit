from .CleavagePattern import CleavagePattern

patterns = [
    CleavagePattern(
        name="any single bond cleavage (acyclic only)",
        smirks="[*:1]-!@[*:2]>>[*:1][*].[*:2][*]"
    ),
]
