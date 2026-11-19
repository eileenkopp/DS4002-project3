"""
"Runs the interactive rock paper scissors game. Uses computer's default webcam, players follow displayed 
directions in the game to play against a random computer opponent. 
"""

import cv2
import time

from game_logic import (
    get_computer_move,
    determine_winner,
    GameScore
)

#TODO: add instructions / welcome screen? 

def main():
    # Open default webcam
    cap = cv2.VideoCapture(0)

    # Check that the camera opened
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

    # Countdown information
    countdown_start = None

    # Temporary simulated gesture
    mock_move = "rock"

    while True:
        # Read one frame from webcam
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Flip horizontally for a mirror-like view
        frame = cv2.flip(frame, 1)

        # Get frame dimensions
        height, width = frame.shape[:2]

        # Define square region in the center
        box_size = int(min(width, height) * 0.65)

        x1 = (width - box_size) // 2
        y1 = (height - box_size) // 2

        x2 = x1 + box_size
        y2 = y1 + box_size

        # Crop the hand region
        hand_crop = frame[y1:y2, x1:x2].copy()

        # Draw green hand-positioning rectangle
        cv2.rectangle(frame,(x1, y1),(x2, y2),(0, 255, 0),2)

        if state == "READY": 
            cv2.putText(frame,"Press SPACE to start!",(20, 80),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255, 255, 255),2)

        if state == "COUNTDOWN":
            elapsed = time.monotonic() - countdown_start

            if elapsed < 3:
                number = 3 - int(elapsed)

                cv2.putText(frame, str(number), (width // 2 - 30, 80), cv2.FONT_HERSHEY_SIMPLEX,2,(0, 255, 255),4)

            else:
                state = "CAPTURE"
        if state == "CAPTURE":
            # TODO: Replace with CNN prediction later
            player_move = mock_move

            # Random computer move
            #computer_move = get_computer_move()

            # Determine winner
            winner = determine_winner(player_move, computer_move)

            # Update score
            score.update(winner)

            # Move to results screen
            state = "RESULT"

        if state == "RESULT":
            cv2.putText(frame, f"You: {player_move.upper()}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255),2)
            cv2.putText(frame,f"Computer: {computer_move.upper()}",(20, 120),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255, 255, 255),2)
            if winner == "tie":
                result_text = "IT'S A TIE!"
            elif winner == "player":
                result_text = "YOU WIN!"
            else:
                result_text = "COMPUTER WINS!"

            cv2.putText(frame, result_text,(20, 170),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 255),3)

        cv2.putText(frame,"SPACE: Play | R: Reset | Q: Quit",(20, height - 20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255, 255, 255),2)
        score_text = (f"You: {score.player} | "f"Computer: {score.computer} | "f"Ties: {score.ties}")
        cv2.putText(frame,score_text,(20, height - 60),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255, 255, 255),2)

        # Display the frame
        cv2.imshow("Rock Paper Scissors", frame)
        cv2.putText(frame,f"Mock gesture: {mock_move.upper()} (1/2/3)", (20, height - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255, 255, 255),2)
        key = cv2.waitKey(1) & 0xFF

        # Quit
        if key == ord("q"):
            break

        # Start a new round
        elif key == 32 and state in ("READY", "RESULT"):
            state = "COUNTDOWN"
            countdown_start = time.monotonic()

            # Computer selects its move before the round
            computer_move = get_computer_move()

        # Reset scores
        elif key == ord("r"):
            score.reset()
            state = "READY"
            player_move = None
            computer_move = None
            winner = None

        # Simulate gesture (temporary)
        elif key == ord("1"):
            mock_move = "rock"

        elif key == ord("2"):
            mock_move = "paper"

        elif key == ord("3"):
            mock_move = "scissors"

    # Release camera and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()