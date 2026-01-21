def sensitivity_analysis(T_data, Y_data, learning_rate, steps, num_ace_samples,rho_assumptions):
    """
    Runs the full sensitivity analysis for a given dataset.

    Args:
        T_data (torch.Tensor): The treatment data.
        Y_data (torch.Tensor): The outcome data.
        learning_rate (float): Learning rate for the Adam optimizer.
        steps (int): Number of training steps for each rho value.
        num_ace_samples (int): Number of samples to estimate ACE.
        rho_assumptions (list or torch.Tensor): List of rho values to assume for sensitivity analysis.

    Returns:
        list: A list of estimated ACE values corresponding to each rho_assumption.
    """
    # Sensitivity Analysis over different rho assumptions
    ace_results = []

    # Loop over different rho assumptions
    for rho_model_assumption in rho_assumptions:
        print(f"\n===== Training model for rho_assumption = {rho_model_assumption.item():.2f} =====")

        # Rational Quadratic Spline for T
        T_transform = T.Spline(input_dim=1, bound=10.0)

        # Rational Quadratic Spline for Y
        Y_transform = T.conditional_spline(input_dim=1, context_dim=1, bound= 10.0) # context_dim = 1 because T is one dimensional

        modules = torch.nn.ModuleList([T_transform, Y_transform])
        optimizer = torch.optim.Adam(modules.parameters(), lr=learning_rate)

        # Training loop
        for step in range(steps):
            optimizer.zero_grad()

            # Inverse transform to get noise and log det Jacobian
            eps_T_pred, log_abs_detJT = T_transform.spline_op(T_data, inverse=True)
            eps_Y_pred, log_abs_detJY = Y_transform.condition(T_data).spline_op(Y_data, inverse=True)

            # Assume a model for the noise distribution
            noise_dist_model = MultivariateNormal(torch.zeros(2), torch.tensor([[1.0, rho_model_assumption],[rho_model_assumption, 1.0]]))

            # Loss calculation
            ln_p_Y_T = noise_dist_model.log_prob(torch.cat([eps_T_pred, eps_Y_pred], dim=1)) + log_abs_detJY + log_abs_detJT
            loss = -ln_p_Y_T.mean()

            loss.backward()
            optimizer.step()

            T_transform.clear_cache()
            Y_transform.condition(T_data).clear_cache()

            if step % 1000 == 0:
                print(f"Step {step} : loss = {loss.item()}")


        # Sample from the noise distribution to estimate ACE
        noise_dist = MultivariateNormal(torch.zeros(2), torch.tensor([[1.0, rho_model_assumption],[rho_model_assumption, 1.0]]))
        noise_samples = noise_dist.sample(torch.Size([num_ace_samples])) # 100000 samples of (eps_T, eps_Y)
        epsT_samples = noise_samples[:, 0].unsqueeze(1)
        epsY_samples = noise_samples[:, 1].unsqueeze(1)
        #T = T_transform.spline_op(epsT_samples)[0]
        #Y = Y_transform.condition(T).spline_op(epsY_samples)[0]

        # Set T to fixed values (e.g., 0 and 1) and compute E[Y|do(T=t)]
        T_fixed_one = torch.ones(num_ace_samples, 1) # Set T to 1
        T_fixed_zero = torch.zeros(num_ace_samples, 1) # Set T to 0

        # Get corresponding Y samples
        Y_do_T_one = Y_transform.condition(T_fixed_one).spline_op(epsY_samples)[0]
        Y_do_T_zero = Y_transform.condition(T_fixed_zero).spline_op(epsY_samples)[0]

        # Compute ACE
        ace = (Y_do_T_one - Y_do_T_zero).mean()
        ace_results.append(ace.item())
        print(f"--> For rho_assumption={rho_model_assumption.item():.2f}, Estimated ACE: {ace.item():.4f}")

    return ace_results