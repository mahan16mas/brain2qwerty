
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
        (4, 2): ["X", 0.37053, 0.34175, 0.32261],
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
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


# ---------------------------------------------------------
# MODEL CONFIGURATIONS
# ---------------------------------------------------------

hidden_sizes = [512, 1024, 2048]

conv_depths = {
    512:  [2, 4, 8],
    1024: [2, 4, 8],
    2048: [1, 2, 4, 8],
}


# ---------------------------------------------------------
# Y-AXIS SETTINGS
# ---------------------------------------------------------

# Change this to control how fine the y-axis ticks are.
# For example:
#   0.01  -> ticks every 0.01
#   0.005 -> ticks every 0.005
#   0.002 -> ticks every 0.002
y_tick_step = 0.05


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def clean(values):
    """Convert missing experiments marked with 'X' to NaN."""
    return np.array([
        np.nan if value == "X" else float(value)
        for value in values
    ])


def format_params(n):
    """Format parameter counts nicely."""
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

fig, axes = plt.subplots(
    1,
    3,
    figsize=(13, 4.5),
    sharey=True
)


# ---------------------------------------------------------
# PLOT
# ---------------------------------------------------------

for ax, hidden_size in zip(axes, hidden_sizes):

    depths = conv_depths[hidden_size]

    errors_42 = clean(errors[hidden_size][(4, 2)])
    errors_21 = clean(errors[hidden_size][(2, 1)])

    # -----------------------------------------------------
    # Transformer (4, 2)
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

    # -----------------------------------------------------
    # Transformer (2, 1)
    # -----------------------------------------------------

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
    # DASHED CONNECTIONS BETWEEN TRANSFORMER CONFIGS
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
    # X-AXIS TICKS
    #
    # First line: conv depth
    # Second line: conv parameter count
    # -----------------------------------------------------

    tick_labels = [
        f'{depth}\n'
        f'{format_params(conv_params[hidden_size][depth])}'
        for depth in depths
    ]

    ax.set_xticks(depths)
    ax.set_xticklabels(tick_labels)

    # -----------------------------------------------------
    # Y-AXIS TICKS
    # -----------------------------------------------------

    # Finer y-axis tick spacing
    ax.yaxis.set_major_locator(
        MultipleLocator(y_tick_step)
    )

    # IMPORTANT:
    # sharey=True normally hides the y tick labels on
    # subplots 2 and 3. This turns them back on.
    ax.tick_params(
        axis='y',
        labelleft=True
    )

    # -----------------------------------------------------
    # TITLES / LABELS
    # -----------------------------------------------------

    ax.set_title(
        f'Conv hidden size = {hidden_size}'
    )

    ax.set_xlabel(
        'Conv depth\nConv parameters'
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
        title='Transformer config / params',
        frameon=False,
        fontsize=8,
        title_fontsize=8
    )


# ---------------------------------------------------------
# Y-AXIS LABEL
# ---------------------------------------------------------

axes[0].set_ylabel('Error rate')


# ---------------------------------------------------------
# FINAL LAYOUT
# ---------------------------------------------------------

plt.tight_layout()
plt.show()