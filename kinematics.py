import numpy as np

class DHLink:
    def __init__(self, a, alpha, d, theta_offset=0.0):
        self.a = a
        self.alpha = alpha
        self.d = d
        self.theta_offset = theta_offset

    def transform(self, theta):
        t = theta + self.theta_offset
        ct, st = np.cos(t), np.sin(t)
        ca, sa = np.cos(self.alpha), np.sin(self.alpha)
        return np.array([
            [ct, -st*ca, st*sa, self.a*ct],
            [st, ct*ca, -ct*sa, self.a*st],
            [0, sa, ca, self.d],
            [0, 0, 0, 1]
        ])

class ArmKinematics:
    def __init__(self, links):
        self.links = links
        self.n = len(links)

    def forward_kinematics(self, thetas):
        T = np.eye(4)
        transforms = [T.copy()]
        for link, theta in zip(self.links, thetas):
            T = T @ link.transform(theta)
            transforms.append(T.copy())
        return T, transforms

    def end_effector_position(self, thetas):
        T, _ = self.forward_kinematics(thetas)
        return T[:3, 3]

    def jacobian(self, thetas, epsilon=1e-6):
        J = np.zeros((3, self.n))
        p0 = self.end_effector_position(thetas)
        for i in range(self.n):
            perturbed = thetas.copy()
            perturbed[i] += epsilon
            p1 = self.end_effector_position(perturbed)
            J[:, i] = (p1 - p0) / epsilon
        return J

    def inverse_kinematics(self, target_pos, initial_thetas, max_iters=200, tol=1e-4, alpha=0.5):
        thetas = np.array(initial_thetas, dtype=float)
        for i in range(max_iters):
            current_pos = self.end_effector_position(thetas)
            error = target_pos - current_pos
            if np.linalg.norm(error) < tol:
                return thetas, True, i
            J = self.jacobian(thetas)
            lam = 0.01
            J_pinv = J.T @ np.linalg.inv(J @ J.T + lam**2 * np.eye(3))
            thetas += alpha * (J_pinv @ error)
        return thetas, False, max_iters
