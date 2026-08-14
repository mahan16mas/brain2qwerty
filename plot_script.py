
errors = {
    512: {
        # actually (2,1)
        (4, 2): [0.7863, "X", "X"],
        #       [ [X] , [] , [ ] ]
        
        # actually (1, 1)
        (2, 1): [0.65051, "X", "X"],
        #       [ [X] , [] , [ ] ]
    },

    1024: {
        (4, 2): [0.76263, 0.74586, 0.77072],
        #       [ [X] , [X] , [X] ]
        (2, 1): [0.45005, 0.77391, 0.45579],
        #       [ [X] , [X] , [X] ]
        # (1,1) d1024 = 0.75475
    },

    # Now this one has FOUR values, corresponding to [1, 2, 4, 8]
    2048: {
        (4, 2): [0.39903, 0.37053, 0.34175, 0.32261],
        (2, 1): [0.7973, 0.45397, 0.75808, "X"],
    },

}




# ---------------------------------------------------------
# PARAMETER COUNTS
# Replace these example values with your actual values
# ---------------------------------------------------------

conv_params = {
    512: {
        2: 1_600_000,
        4: 3_200_000,
        8: 6_400_000,
    },

    1024: {
        2: 4_800_000,
        4: 12_000_000,
        8: 23_700_000,
    },

    2048: {
        1: 3_200_000,
        2: 16_000_000,
        4: 41_000_000,
        8: 91_300_000,
    },
}


transformer_params = {
    512: {
        (4, 2): 12_500_000,
        (2, 1): 5_700_000,
    },

    1024: {
        (4, 2): 50_300_000,
        (2, 1): 23_000_000,
    },

    2048: {
        (4, 2): 201_000_000,
        (2, 1): 92_000_000,
    },
}


# Replace these with your actual values
CEBRA_error = 0.44308
CEBRA_noisy_error = 0.23203
show_CEBRA_noisy = False


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

# True  -> only 1024 and 2048
# False -> 512, 1024, and 2048
ignore_512 = True

if ignore_512:
    hidden_sizes = [1024, 2048]
else:
    hidden_sizes = [512, 1024, 2048]


conv_depths = {
    512:  [2, 4, 8],
    1024: [2, 4, 8],
    2048: [1, 2, 4, 8],
}

# Fine y-axis tick spacing
y_tick_step = 0.05


# ---------------------------------------------------------
# BASELINE ERROR RATES
# ---------------------------------------------------------


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def clean(values):
    return np.array([
        np.nan if value == "X" else float(value)
        for value in values
    ])


def format_params(n):
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    elif n >= 1e6:
        return f"{n / 1e6:.1f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.1f}K"
    else:
        return str(n)


# ---------------------------------------------------------
# CREATE FIGURE
# ---------------------------------------------------------

n_plots = len(hidden_sizes)

fig, axes = plt.subplots(
    1,
    n_plots,
    figsize=(4.5 * n_plots, 4.5),
    sharey=True
)

axes = np.atleast_1d(axes)


# ---------------------------------------------------------
# PLOT
# ---------------------------------------------------------

for ax, hidden_size in zip(axes, hidden_sizes):

    depths = conv_depths[hidden_size]

    errors_42 = clean(errors[hidden_size][(4, 2)])
    errors_21 = clean(errors[hidden_size][(2, 1)])

    # -----------------------------------------------------
    # YOUR MODELS
    # -----------------------------------------------------

    ax.plot(
        depths,
        errors_42,
        marker='o',
        markersize=7,
        linewidth=2,
        label=(
            f'(4, 2) — '
            f'{format_params(transformer_params[hidden_size][(4, 2)])}'
        )
    )

    ax.plot(
        depths,
        errors_21,
        marker='s',
        markersize=7,
        linewidth=2,
        label=(
            f'(2, 1) — '
            f'{format_params(transformer_params[hidden_size][(2, 1)])}'
        )
    )

    # -----------------------------------------------------
    # CONNECTIONS BETWEEN TRANSFORMER CONFIGS
    # -----------------------------------------------------

    for x, y_42, y_21 in zip(
        depths,
        errors_42,
        errors_21
    ):
        if not np.isnan(y_42) and not np.isnan(y_21):
            ax.plot(
                [x, x],
                [y_42, y_21],
                linestyle='--',
                linewidth=1,
                alpha=0.6
            )

    # -----------------------------------------------------
    # BASELINE MODELS
    # -----------------------------------------------------

    ax.axhline(
        y=CEBRA_error,
        linestyle='-.',
        linewidth=1.5,
        label=f'CEBRA ({CEBRA_error:.3f})'
    )

    # CEBRA_noisy baseline
    if show_CEBRA_noisy:
        ax.axhline(
            y=CEBRA_noisy_error,
            linestyle=':',
            linewidth=2,
            label=f'CEBRA_noisy ({CEBRA_noisy_error:.3f})'
        )

    # -----------------------------------------------------
    # X-AXIS
    # -----------------------------------------------------

    tick_labels = [
        f'{depth}\n'
        f'{format_params(conv_params[hidden_size][depth])}'
        for depth in depths
    ]

    ax.set_xticks(depths)
    ax.set_xticklabels(tick_labels)

    ax.set_xlabel(
        'Conv depth\nConv parameters'
    )

    # -----------------------------------------------------
    # Y-AXIS
    # -----------------------------------------------------

    ax.yaxis.set_major_locator(
        MultipleLocator(y_tick_step)
    )

    # Show y tick values on every subplot
    ax.tick_params(
        axis='y',
        labelleft=True
    )

    # -----------------------------------------------------
    # TITLE
    # -----------------------------------------------------

    ax.set_title(
        f'Conv hidden size = {hidden_size}'
    )

    # -----------------------------------------------------
    # GRID
    # -----------------------------------------------------

    ax.grid(
        axis='y',
        linestyle=':',
        alpha=0.4
    )

    # -----------------------------------------------------
    # LEGEND
    # -----------------------------------------------------

    ax.legend(
        title='Model / Transformer params',
        frameon=False,
        fontsize=8,
        title_fontsize=8
    )


# ---------------------------------------------------------
# SHARED Y LABEL
# ---------------------------------------------------------

axes[0].set_ylabel('Error rate')


# ---------------------------------------------------------
# FINAL LAYOUT
# ---------------------------------------------------------

plt.tight_layout()
plt.show()