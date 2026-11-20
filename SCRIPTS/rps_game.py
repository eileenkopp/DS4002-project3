
"""
Interactive rock-paper-scissors game using a trained CNN
to classify the player's hand gesture.
"""

import cv2
import time

from game_logic import (
    get_computer_move,
    determine_winner,
    GameScore
)

from gesture_classifier import (
    load_classifier,
    predict_gesture
)

def draw_centered_text(frame, text, y, font_scale=1.0,
                       color=(255, 255, 255), thickness=2):
    """Draw text horizontally centered at a given y-coordinate."""
    height, width = frame.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Calculate the text width and height
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    # Center text horizontally
    x = (width - text_width) // 2

    cv2.putText(
        frame, text, (x, y),
        font, font_scale, color, thickness
    )
    
def main():
    # Load trained CNN once at startup
    print("Loading gesture classifier...")
    model = load_classifier()
    print("Classifier loaded successfully!")

    # Open default webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Game state
    state = "READY"

    # Score tracker
    score = GameScore()

    # Round information
    player_move = None
    computer_move = None
    winner = None
    confidence = None

    # Countdown information
    countdown_start = None

    while True:
        # Read one frame from webcam
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Flip horizontally for mirror-like view
        frame = cv2.flip(frame, 1)

        # Get frame dimensions
        height, width = frame.shape[:2]

        # Define square hand region
        box_size = int(min(width, height) * 0.65)

        x1 = (width - box_size) // 2
        y1 = (height - box_size) // 2
        x2 = x1 + box_size
        y2 = y1 + box_size

        # Crop hand region BEFORE drawing overlays
        hand_crop = frame[y1:y2, x1:x2].copy()

        # Draw positioning rectangle
        cv2.rectangle(
            frame, (x1, y1), (x2, y2),
            (0, 255, 0), 2
        )

        # ---------------- READY ----------------
        if state == "READY":
            draw_centered_text(
                frame, "Press SPACE to start!",
                85, font_scale=1.1, thickness=3
            )

        # ---------------- COUNTDOWN ----------------
        if state == "COUNTDOWN":
            elapsed = time.monotonic() - countdown_start

            if elapsed < 3:
                number = 3 - int(elapsed)

                draw_centered_text(
                    frame, str(number),
                    height // 2,
                    font_scale=3.0,
                    color=(0, 255, 255),
                    thickness=6
                )
            else:
                state = "CAPTURE"

        # ---------------- CAPTURE ----------------
        if state == "CAPTURE":
        
            # Predict player's hand gesture using CNN
            player_move, confidence = predict_gesture(
                model, hand_crop
            )
        
            print(
                f"Predicted: {player_move} "
                f"(confidence: {confidence:.2%})"
                )
        
            # Determine winner
            winner = determine_winner(player_move, computer_move)
        
            # Update score
            score.update(winner)
        
            # Move to results screen
            state = "RESULT"

        # ---------------- RESULT ----------------
        if state == "RESULT":

            if winner == "tie":
                result_text = "IT'S A TIE!"
                result_color = (0, 255, 255)

            elif winner == "player":
                result_text = "YOU WIN!"
                result_color = (0, 255, 0)

            else:
                result_text = "COMPUTER WINS!"
                result_color = (0, 0, 255)

            # Large centered result
            draw_centered_text(
                frame, result_text,
                90,
                font_scale=1.5,
                color=result_color,
                thickness=4
            )

            # Player and computer moves
            draw_centered_text(
                frame,
                f"You: {player_move.upper()} ({confidence:.0%})",
                140,
                font_scale=0.9,
                thickness=2
            )

            draw_centered_text(
                frame,
                f"Computer: {computer_move.upper()}",
                180,
                font_scale=0.9,
                thickness=2
            )

        # ---------------- SCOREBOARD ----------------
        score_text = (
            f"You: {score.player}   |   "
            f"Computer: {score.computer}   |   "
            f"Ties: {score.ties}"
        )

        draw_centered_text(
            frame, score_text,
            height - 65,
            font_scale=0.9,
            color=(0, 255, 255),
            thickness=3
        )

        # ---------------- CONTROLS ----------------
        draw_centered_text(
            frame,
            "SPACE: Play  |  R: Reset  |  Q: Quit",
            height - 20,
            font_scale=0.7,
            thickness=2
        )
        
        # Display completed frame
        cv2.imshow("Rock Paper Scissors", frame)

        key = cv2.waitKey(1) & 0xFF

        # Quit
        if key == ord("q"):
            break

        # Start a new round
        elif key == 32 and state in ("READY", "RESULT"):
            state = "COUNTDOWN"
            countdown_start = time.monotonic()

            # Computer chooses move before round
            computer_move = get_computer_move()

        # Reset scores
        elif key == ord("r"):
            score.reset()
            state = "READY"
            player_move = None
            computer_move = None
            winner = None
            confidence = None

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
