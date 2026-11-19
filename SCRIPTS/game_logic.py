"""
Implements the rock paper scissors logic needed in rps_game.py
"""
import random

MOVES = ["rock", "paper", "scissors"]


def get_computer_move():
    return random.choice(MOVES)

def determine_winner(player_move, computer_move):
    # Validate both moves
    if player_move not in MOVES or computer_move not in MOVES:
        raise ValueError("Invalid rock-paper-scissors move")

    # Case 1: Tie
    if player_move == computer_move:
        return "tie"

    # Case 2: Player wins
    winning_combinations = {
        ("rock", "scissors"),
        ("paper", "rock"),
        ("scissors", "paper")
    }

    if (player_move, computer_move) in winning_combinations:
        return "player"

    # Case 3: Computer wins
    return "computer"

class GameScore:
    def __init__(self):
        self.player = 0
        self.computer = 0
        self.ties = 0

    def update(self, winner):
        if winner == "player":
            self.player += 1
        elif winner == "computer":
            self.computer += 1
        elif winner == "tie":
            self.ties += 1
        else:
            raise ValueError("Invalid round result")

    def reset(self):
        self.player = 0
        self.computer = 0
        self.ties = 0