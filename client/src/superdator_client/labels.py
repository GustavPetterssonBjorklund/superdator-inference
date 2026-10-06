"""Labels the inference model can detect.

The `name` field of each detection returned by the server is one of LABELS.values().
"""

LABELS: dict[int, str] = {
    0: "gear_32269",
    1: "cog_32072",
    2: "joint_62520c01",
    3: "beam-1x3_32523",
    4: "axle-pinhole_32034",
    5: "pin_4274",
    6: "wheel_13971",
    7: "torse_973px431c01",
}
