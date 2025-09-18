extends Node
class_name Player

const MathUtils := preload("res://src/util/math_utils.gd")

var display_name: String
var experience: int

func _init(name: String, xp: int) -> void:
    display_name = name
    experience = xp

func level_up(delta: int) -> void:
    experience += delta
    var level := MathUtils.calculate_level(experience)
    var bonus := MathUtils.calculate_power_level(level)
    _log_progress(level, bonus)

func _log_progress(level: int, bonus: int) -> void:
    print("%s reached level %d with bonus %d" % [display_name, level, bonus])
