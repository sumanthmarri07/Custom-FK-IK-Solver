import numpy as np
import matplotlib.pyplot as plt
from kinematics import DHLink, ArmKinematics

arm = ArmKinematics([DHLink(0.3,0,0), DHLink(0.25,0,0), DHLink(0.15,0,0)])
_, transforms = arm.forward_kinematics(np.radians([30,45,-20]))
xs, ys = [T[0,3] for T in transforms], [T[1,3] for T in transforms]
plt.plot(xs, ys, '-o'); plt.grid(True); plt.axis('equal'); plt.savefig("arm.png")
print("Plot saved as arm.png")
