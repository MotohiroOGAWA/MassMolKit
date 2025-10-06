from .CleavagePattern import CleavagePattern

HETERO = "#8,#7,#16,#15,#9,#17,#35,#53"  # O,N,S,P,F,Cl,Br,I (heteroatoms)
X = "#9,#17,#35,#53"  # F,Cl,Br,I (halogens)

patterns = [
    CleavagePattern(
        name="any single bond cleavage (acyclic only)",
        smirks="[*:1]-!@[*:2]>>[*:1][*].[*:2][*]"
    ),

    # # --- Remote hydrogen rearrangement (type a: A-B-XH → A=B + HX) ---
    # CleavagePattern(
    #     name="remote H rearrangement (A-B-XH → A=B + HX)",
    #     smirks=f"[!#1;!{X}]-[*:1]-[*:2]-[{HETERO}:3]-[H]>>[*:1]=[*:2]"
    #     # smirks=f"[*:1]-[*:2]-[{X}:3]-[H]>>[*:1]=[*:2].[{X}:3][H]"
    # ),

    # # --- Remote hydrogen rearrangement (type b: C-H-DX → C-H + D=X) ---
    # CleavagePattern(
    #     name="remote H rearrangement (C-H-DX → C-H + D=X)",
    #     smirks=f"[*:1]-[*:2]-[{HETERO}:3]-[H]>>[*:1][H]"
    #     # smirks=f"[*:1]-[H]-[*:2]-[{X}:3]>>[*:1][H].[*:2]=[{X}:3]"
    # ),

    # # --- Retro-Diels–Alder (RDA) ---
    # CleavagePattern(
    #     name="retro-Diels–Alder (RDA) reaction",
    #     # 6-membered ring with two conjugated double bonds, breaking into two fragments
    #     smirks="[*:1]1=[:2][*:3][*:4][*:5][*:6]1>>[*:6]=[*:1][*:2]=[*:3]" 
    #     # smirks="[*:1]1=[:2][*:3][*:4][*:5][*:6]1>>[*:6]=[*:1][*:2]=[*:3].[*:4]=[*:5]" 
    # ),

    # # --- Retro-ene reaction ---
    # CleavagePattern(
    #     name="retro-ene reaction",
    #     # ene reaction: allylic H shift + double bond migration
    #     smirks="[*:1]=[*:2]-[*:3]-[*:4]-[*:5H]>>[*:1]([H])-[*:2]=[*:3]"
    #     # smirks="[*:1]=[*:2]-[*:3]-[*:4]-[*:5H]>>[*:1]([H])-[*:2]=[*:3].[*:4]=[*:5]"
    # ),
]
