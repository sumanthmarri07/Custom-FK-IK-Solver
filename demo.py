import numpy as np
from kinematics import DHLink, ArmKinematics

links = [DHLink(0.3,0,0), DHLink(0.25,0,0), DHLink(0.15,0,0)]
arm = ArmKinematics(links)

thetas = np.radians([30, 45, -20])
pos = arm.end_effector_position(thetas)
print(f"FK Position: {pos}")

target = np.array([0.4, 0.3, 0.0])
solution, converged, iters = arm.inverse_kinematics(target, np.zeros(3))
print(f"IK Converged: {converged} | Solution: {np.degrees(solution)}")
