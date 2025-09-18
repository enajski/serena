extends Resource
class_name MathUtils

static func calculate_level(total_experience: int) -> int:
    if total_experience < 0:
        return 0
    return int(total_experience / 100)

static func calculate_power_level(level: int) -> int:
    return level * level + 4
