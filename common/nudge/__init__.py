"""
NUDGE -- Neural Unified Direction and Gap Estimation.

Learns discrete spatial corrections (x, y, z) from camera images + target
bounding box masks. Self-supervised from successful pick/place approaches:
the ground truth at each frame is the discretized vector from the gripper's
current position to the final successful grasp/place position.

Output: 3 classification heads, each predicting one of 7 classes
{-3, -2, -1, 0, +1, +2, +3} representing correction magnitudes per axis.
Class 0 = aligned (<3mm); all zeros = "done, release".
"""
