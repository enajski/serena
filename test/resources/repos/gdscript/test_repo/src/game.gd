extends Node
class_name Game

const STARTING_EXPERIENCE := 120

var _player: Player

func _ready() -> void:
    _player = Player.new("Ari", STARTING_EXPERIENCE)
    _player.level_up(STARTING_EXPERIENCE)

func recruit_player(name: String, experience: int) -> Player:
    var recruit := Player.new(name, experience)
    recruit.level_up(experience)
    return recruit

func total_power(level: int) -> int:
    return MathUtils.calculate_power_level(level) + level
