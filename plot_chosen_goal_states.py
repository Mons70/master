import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import colorcet as cc
import numpy as np

# Original 18 goals
keyframes = {
        "home": [-0.125, -0.625, 0.25],
        1: [-0.25, -1.0, 0.25],
        2: [-0.352465, -1.59862, 0.25],
        3: [-0.521426, -2.13349, 0.25],
        4: [-1.01012, -2.46729, 0.25],
        5: [-2.01529, -2.4747, 0.25],
        6: [-2.70571, -2.30555, 0.25],
        7: [-3.33213, -1.54147, 0.25],
        8: [-4.3052, -1.01978, 0.25],
        9: [-4.65979, -0.304562, 0.25],
        10: [-4.74108, 0.688841, 0.282658],
        11: [-4.72309, 2.03176, 0.332065],
        12: [-4.261, 2.18683, 0.482171],
        13: [-3.82627, 2.35773, 0.664309],
        14: [-3.42159, 2.52583, 0.74358],
        15: [-2.97588, 2.51519, 0.668477],
        16: [-2.12638, 2.40482, 0.294929],
        17: [-0.520779, 1.66374, 0.294929],
        18: [-0.0844852, 0.465219, 0.294929],
        19: [1.93695, -0.13810, 0.45],
        20: [-4.98318, -3.95614, 0.25],
        21: [0.79131, -4.68012, 0.25],
        22: [-0.02520, 3.68893, 0.9],
        23: [3.74955, -1.58986, 0.27],
        24: [4.48456, 2.75455, 0.30],
        25: [4.58215, -3.42454, 0.25],
        26: [2.43022, 4.54289, 0.25],
        27: [1.12297, 1.19466, 0.25],
        28: [-3.49162, -0.55558, 0.25],
        29: [-0.94072, 4.81276, 0.25],
        30: [2.14062, 0.77786, 0.92],
        31: [-1.30249, 0.09238, 0.47],
        32: [2.50175, 3.22686, 0.25],
        33: [4.69978, -4.66257, 0.25],
        34: [1.22319, -2.47120, 0.25],
        35: [-3.12424, 0.40762, 0.55],
        36: [4.77988, -0.56363, 0.25],
        37: [-3.55986, 4.69313, 0.25],
        38: [0.28912, -4.63385, 0.25],
        39: [1.65215, 4.04531, 0.33],
        40: [-0.87880, 3.33669, 0.25]
    }

# Extract coordinates
goal_states = ['home',1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
# goal_states = [1,2,8,11,14,17]
goal_coords = [keyframes.get(key) for key in goal_states]
# new_x, new_y = zip(*new_goals)

# Plot
matplotlib.use('pgf')
matplotlib.rcParams.update(
    {
        'pgf.texsystem': 'pdflatex',
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    }
)

# Load your image (as array)
image = plt.imread("/home/mons/dev/private/thesis-paper/figures/fractal_noise.png")  # shape: (H, W, 3) or (H, W)
height, width = image.shape[:2]

# Set the extent you want (just like imshow)
xmin, xmax = -5, 5
ymin, ymax = -5, 5

# Generate coordinate grid with +1 size
x = np.linspace(xmin, xmax, width + 1)
y = np.linspace(ymin, ymax, height + 1)

fig, ax = plt.subplots()

# Plot background with grayscale image
# ax.pcolormesh(x, y, np.flipud(image), cmap='gray', shading='auto')
# plt.figure()
colors = cc.glasbey_light[:len(goal_states)]
for i,(goal, coords) in enumerate(zip(goal_states, goal_coords)):
    print(i)
    ax.scatter(coords[0], coords[1], color=colors[i], label=f'Goal {goal}', marker='x')
# plt.scatter(new_x, new_y, color='red', label='New Goals')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_title('Top-down View of chosen goal states')
# img = mpimg.imread('/home/mons/dev/private/thesis-paper/figures/fractal_noise.png')
# plt.imshow(img, extent=[-5, 5, -5, 5], origin='upper', cmap='gray')
ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), title="Goal States")
ax.grid(True)
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
fig.tight_layout()
# plt.show()
plt.savefig('/home/mons/dev/private/thesis-paper/figures/all_goal_states.pgf')
