import numpy as np
import matplotlib.pyplot as plt


def plot_target_layout(settings, state):
        pattern_types = settings['pattern_type']
        
        for idx in range(len(pattern_types)):
            objp = state[f'objp_{idx}']

            min_alpha = 0.15
            alpha = min_alpha + (1-min_alpha) * (len(objp) - np.arange(len(objp))) / len(objp)
            alpha[-1] = 1
            plt.scatter(objp[:, 0], objp[:, 1], label=f'{idx}', alpha=alpha)

        plt.axis('equal')
        plt.legend()
        plt.title('Pattern Layout')
        plt.gca().invert_yaxis()
        plt.show()