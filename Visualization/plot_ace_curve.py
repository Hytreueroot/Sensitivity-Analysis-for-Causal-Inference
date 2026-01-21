def plot_ace_curve(rho_assumptions, ace_results, true_alpha, title, true_rho=None):
    """
    Plots the ACE curve from the results of a sensitivity analysis.

    Args:
        rho_assumptions (torch.Tensor): The tested rho values (x-axis).
        ace_results (list): The estimated ACE values (y-axis).
        true_alpha (float): The true causal effect for the reference line.
        title (str): The title for the plot.
        true_rho (float): The true confounding for the reference line.
    """
    # Calculate min and max ACE for shading
    min_ace = min(ace_results)
    max_ace = max(ace_results)

     # Plotting the ACE curve
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))
    # Draw ACE curve
    plt.plot(rho_assumptions.numpy(), ace_results, marker='o', linestyle='-', color='darkslateblue', zorder=10)
    # Intervals
    plt.fill_between(rho_assumptions.numpy(), min_ace, max_ace, color='skyblue', alpha=0.3, label=f'Estimated ACE Bounds [{min_ace:.2f}, {max_ace:.2f}]')


    # Reference lines for true ACE and true rho
    plt.axhline(y=true_alpha, color='tomato', linestyle='--', label=f'True ACE (alpha={true_alpha})')

    # Only plot true_rho line if true_rho is provided
    if true_rho is not None:
        plt.axvline(x=true_rho, color='forestgreen', linestyle='--', label=f'True Confounding (rho={true_rho})')

    plt.title(title, fontsize=16)
    plt.xlabel("Assumed Confounding (rho_model_assumption)", fontsize=12)
    plt.ylabel("Estimated ACE", fontsize=12)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.grid(True)
    plt.show()