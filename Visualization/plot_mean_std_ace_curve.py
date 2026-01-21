def plot_mean_std_ace_curve(rho_assumptions, ace_means, ace_stds, true_alpha, title, num_runs, true_rho=None):
    """
    Plots the mean ACE curve with standard deviation shading from multiple runs of sensitivity analysis.

    Args:
        rho_assumptions (torch.Tensor): The tested rho values (x-axis).
        ace_means (list): The mean estimated ACE values (y-axis).
        ace_stds (list): The standard deviation of estimated ACE values.
        true_alpha (float): The true causal effect for the reference line.
        title (str): The title for the plot.
        num_runs (int): Number of runs
        true_rho (float): The true confounding for the reference line.
    """
    lower_bound = ace_means - ace_stds
    upper_bound = ace_means + ace_stds

    # Plotting the mean ACE curve with std shading
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))

    # Draw mean ACE curve
    plt.plot(rho_assumptions, ace_means, marker='o', linestyle='-', color='darkslateblue', zorder=10, linewidth=2, markersize=6, label="Mean ACE (over {} runs)".format(num_runs))

    # Shaded area for std deviation
    plt.fill_between(rho_assumptions, lower_bound, upper_bound, color='skyblue', alpha=0.3, label='Standard Deviation')

    # Reference lines for true ACE and true rho
    plt.axhline(y=true_alpha, color='tomato', linestyle='--', linewidth=2, label=f'True ACE (alpha={true_alpha})')

    # Only plot true_rho line if true_rho is provided
    if true_rho is not None:
        plt.axvline(x=true_rho, color='forestgreen', linestyle='--', linewidth=2, label=f'True Confounding (rho={true_rho})')

    plt.title(title, fontsize=20, fontweight='bold', pad=20)
    plt.xlabel("Assumed Confounding", fontsize=14, fontweight='bold')
    plt.ylabel("Estimated ACE", fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='best', fontsize=12, frameon=True, framealpha=0.9, shadow=True)
    plt.tight_layout()
    plt.grid(True, alpha=0.5)
    plt.show()