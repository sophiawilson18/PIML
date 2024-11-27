import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy 

class PINNs:
    def __init__(self, X_colloc, net_transform, net_pde_user, loss_f, layers, lr, param_pde=None,
                 type_problem='forward', type_formulation='strong', thres=None,
                 X_bc=None, u_bc=None, net_bc=None, X_init=None, u_init=None, net_init=None, X_data=None, u_data=None,
                 X_other=None, u_other=None, net_other=None, X_test=None, u_test=None, X_traction=None, w_pde=1, model_init=None):
        """
        Initialization function for the PINNs class in PyTorch.
        """

        # Process boundary condition data
        if X_bc is None:
            self.X_bc, self.u_bc, self.nb_bc = None, 0, 0
            print("No data on the boundary")
        else:
            self.X_bc = torch.tensor(X_bc, dtype=torch.float64)
            self.u_bc = torch.tensor(u_bc, dtype=torch.float64)
            self.nb_bc = self.X_bc.shape[0]
            self.net_bc = net_bc if net_bc is not None else net_transform

        # Process initial condition data
        if X_init is None:
            self.X_init, self.u_init, self.nb_init = None, 0, 0
            print("No data at the initial instant")
        else:
            self.X_init = torch.tensor(X_init, dtype=torch.float64)
            self.u_init = torch.tensor(u_init, dtype=torch.float64)
            self.nb_init = self.X_init.shape[0]
            self.net_init = net_init if net_init is not None else net_transform

        # Process data inside the domain
        if X_data is None:
            self.X_data, self.u_data, self.nb_data = None, 0, 0
            print("No data inside the domain")
        else:
            self.X_data = torch.tensor(X_data, dtype=torch.float64)
            self.u_data = torch.tensor(u_data, dtype=torch.float64)
            self.nb_data = self.X_data.shape[0]

        # Process other condition data
        if X_other is None:
            self.X_other, self.u_other, self.nb_other = None, 0, 0
            print("No other condition is provided")
        else:
            self.X_other = torch.tensor(X_other, dtype=torch.float64)
            self.u_other = torch.tensor(u_other, dtype=torch.float64)
            self.nb_other = self.X_other.shape[0]
            self.net_other = net_other if net_other is not None else net_transform

        # Process testing data
        if X_test is None:
            self.X_test, self.u_test, self.nb_test = None, 0, 0
            print("No data for testing")
        else:
            self.X_test = torch.tensor(X_test, dtype=torch.float64)
            self.u_test = torch.tensor(u_test, dtype=torch.float64)
            self.nb_test = self.X_test.shape[0]

        # Process traction data
        if X_traction is None:
            self.X_traction, self.nb_traction = None, 0
        else:
            self.X_traction = torch.tensor(X_traction, dtype=torch.float64)
            self.nb_traction = self.X_traction.shape[0]

        self.X_colloc = torch.tensor(X_colloc, dtype=torch.float64)
        self.type_problem = type_problem
        self.type_formulation = type_formulation
        self.param_pde = None
        self.nb_param = 1
        self.param_pde_array = None

        if self.type_problem == 'inverse':
            if param_pde is None:
                raise ValueError("Must provide initial value for the PDE parameters")
            self.param_pde = torch.nn.Parameter(torch.tensor(param_pde, dtype=torch.float64))
            self.nb_param = self.param_pde.shape[0]
        elif param_pde is not None:
            self.param_pde = torch.tensor(param_pde, dtype=torch.float64)
            self.nb_param = self.param_pde.shape[0]

        self.nb_colloc = self.X_colloc.shape[0]
        self.net_pde_user = net_pde_user
        self.loss_f = loss_f
        self.w_pde = w_pde
        self.pde_weights = self.w_pde
        self.model_init = model_init

        self.layers = layers
        self.net_transform = net_transform

        # Define the neural network model
        if self.model_init is None:
            self.net_u = self.build_model(layers, type_formulation)
        else:
            self.net_u = self.load_pretrained_model(model_init)

        self.optimizer = optim.Adam(self.net_u.parameters(), lr=lr)
        self.loss_array = []
        self.test_array = []
        self.thres = thres
        self.epoch = 0

    def build_model(self, layers, type_formulation):
        """
        Build the neural network for PINNs.
        """
        model = nn.Sequential()
        model.add_module("Input", nn.Linear(layers[0], layers[1]))
        for i in range(1, len(layers) - 2):
            model.add_module(f"Hidden_{i}", nn.Linear(layers[i], layers[i + 1]))
            model.add_module(f"Activation_{i}", nn.Tanh())
        model.add_module("Output", nn.Linear(layers[-2], layers[-1]))

        # Initialize weights and biases
        for m in model:
            if isinstance(m, nn.Linear):
                if type_formulation == 'weak':
                    nn.init.normal_(m.weight, mean=0, std=0.1)
                    nn.init.zeros_(m.bias)
                else:
                    nn.init.xavier_normal_(m.weight)
                    nn.init.zeros_(m.bias)
        return model

    def load_pretrained_model(self, model_init):
        """
        Load a pretrained model for PINNs.
        """
        pretrained_model = nn.Sequential()
        for i, layer in enumerate(model_init.layers):
            W = torch.tensor(layer.get_weights()[0], dtype=torch.float64)
            b = torch.tensor(layer.get_weights()[1], dtype=torch.float64)
            new_layer = nn.Linear(W.shape[0], W.shape[1])
            new_layer.weight.data = W
            new_layer.bias.data = b
            pretrained_model.add_module(f"Layer_{i}", new_layer)
        return pretrained_model
    
    def pinns_training_variables(self):
        """
        Define training parameters in the neural networks.

        Returns:
            List of training parameters, including neural network parameters and PDE parameters (if inverse problem).
        """
        var = list(self.net_u.parameters())  # Trainable parameters of the neural network
        if self.type_problem == 'inverse':
            var.append(self.param_pde)  # Include PDE parameters for inverse problems
        return var

    def net_pde(self, X_f, model_nn, param_f=None, X_traction=None):
        """
        Call PDE function defined by users.

        Args:
            X_f (torch.Tensor): Collocation points.
            model_nn (torch.nn.Module): Neural network model.
            param_f (torch.Tensor, optional): Parameter of the PDE. Defaults to None.
            X_traction (torch.Tensor, optional): Traction points. Defaults to None.

        Returns:
            torch.Tensor: PDE residual vectors.
        """
        if self.type_problem == 'inverse':
            if X_traction is None:
                f = self.net_pde_user(X_f, model_nn, param_f)
            else:
                f = self.net_pde_user(X_f, model_nn, X_traction, param_f)
        elif self.type_problem == 'forward' and self.type_formulation == 'weak':
            if X_traction is None:
                f = self.net_pde_user(X_f, model_nn)
            else:
                f = self.net_pde_user(X_f, model_nn, X_traction)
        else:
            f = self.net_pde_user(X_f, model_nn)
        return f

    def loss_pinns(self, X_f, param_f, model_nn, u_pred_bc, u_star_bc, u_pred_init, u_star_init, u_pred_data,
                u_star_data, u_pred_other, u_star_other, X_traction, pde_weights):
        """
        Define the cost function.

        Args:
            X_f (torch.Tensor): Collocation points.
            param_f (torch.Tensor): Parameter of the PDE.
            model_nn (torch.nn.Module): Neural network model.
            u_pred_bc (torch.Tensor): Prediction for the solution on the boundary.
            u_star_bc (torch.Tensor): Reference solution on the boundary.
            u_pred_init (torch.Tensor): Prediction for the solution at initial instant.
            u_star_init (torch.Tensor): Reference solution at initial instant.
            u_pred_data (torch.Tensor): Prediction for observed measurements.
            u_star_data (torch.Tensor): Reference solution for observed measurements.
            u_pred_other (torch.Tensor): Prediction for the solution on other boundary.
            u_star_other (torch.Tensor): Reference solution on other boundary.
            X_traction (torch.Tensor): Traction points.
            pde_weights (float): Weights for PDE residuals.

        Returns:
            tuple: Loss values during training (total loss, loss_bc, loss_init, loss_data, loss_other, loss_f).
        """
        loss_obs = torch.tensor(0.0, dtype=torch.float64)
        loss_bc = torch.tensor(0.0, dtype=torch.float64)
        loss_init = torch.tensor(0.0, dtype=torch.float64)
        loss_data = torch.tensor(0.0, dtype=torch.float64)
        loss_other = torch.tensor(0.0, dtype=torch.float64)
        loss_f = torch.tensor(0.0, dtype=torch.float64)

        # Compute PDE residuals
        if self.nb_colloc > 0:
            f = self.net_pde(X_f, model_nn, param_f, X_traction)
        else:
            f = torch.tensor(0.0, dtype=torch.float64)

        if self.type_problem != 'generalization':
            loss_f += self.loss_f(f)

            # Boundary condition loss
            if self.nb_bc > 0:
                for i in range(u_star_bc.shape[1]):
                    if not torch.isnan(u_star_bc[0, i]):
                        loss_bc += torch.mean((u_pred_bc[:, i] - u_star_bc[:, i]) ** 2)
                        loss_obs += torch.mean((u_pred_bc[:, i] - u_star_bc[:, i]) ** 2)

            # Initial condition loss
            if self.nb_init > 0:
                for i in range(u_star_init.shape[1]):
                    if not torch.isnan(u_star_init[0, i]):
                        loss_init += torch.mean((u_pred_init[:, i] - u_star_init[:, i]) ** 2)
                        loss_obs += torch.mean((u_pred_init[:, i] - u_star_init[:, i]) ** 2)

            # Data loss
            if self.nb_data > 0:
                for i in range(u_star_data.shape[1]):
                    if not torch.isnan(u_star_data[0, i]):
                        loss_data += torch.mean((u_pred_data[:, i] - u_star_data[:, i]) ** 2)
                        loss_obs += torch.mean((u_pred_data[:, i] - u_star_data[:, i]) ** 2)

            # Other boundary condition loss
            if self.nb_other > 0:
                for i in range(u_star_other.shape[1]):
                    if not torch.isnan(u_star_other[0, i]):
                        loss_other += torch.mean((u_pred_other[:, i] - u_star_other[:, i]) ** 2)
                        loss_obs += torch.mean((u_pred_other[:, i] - u_star_other[:, i]) ** 2)

        else:
            # Generalization case
            for i_param in range(self.nb_param):
                if self.nb_bc > 0:
                    size_bc = u_star_bc.shape[0] // self.nb_param
                    for i in range(u_star_bc.shape[1]):
                        if not torch.isnan(u_star_bc[size_bc * i_param:size_bc * (i_param + 1), i][0]):
                            loss_obs += torch.mean((u_pred_bc[size_bc * i_param:size_bc * (i_param + 1), i] -
                                                    u_star_bc[size_bc * i_param:size_bc * (i_param + 1), i]) ** 2)

                if self.nb_init > 0:
                    size_init = u_star_init.shape[0] // self.nb_param
                    for i in range(u_star_init.shape[1]):
                        if not torch.isnan(u_star_init[size_init * i_param:size_init * (i_param + 1), i][0]):
                            loss_obs += torch.mean((u_pred_init[size_init * i_param:size_init * (i_param + 1), i] -
                                                    u_star_init[size_init * i_param:size_init * (i_param + 1), i]) ** 2)

                if self.nb_data > 0:
                    size_data = u_star_data.shape[0] // self.nb_param
                    for i in range(u_star_data.shape[1]):
                        if not torch.isnan(u_star_data[size_data * i_param:size_data * (i_param + 1), i][0]):
                            loss_obs += torch.mean((u_pred_data[size_data * i_param:size_data * (i_param + 1), i] -
                                                    u_star_data[size_data * i_param:size_data * (i_param + 1), i]) ** 2)

                if self.nb_other > 0:
                    size_other = u_star_other.shape[0] // self.nb_param
                    for i in range(u_star_other.shape[1]):
                        if not torch.isnan(u_star_other[size_other * i_param:size_other * (i_param + 1), i][0]):
                            loss_obs += torch.mean((u_pred_other[size_other * i_param:size_other * (i_param + 1), i] -
                                                    u_star_other[size_other * i_param:size_other * (i_param + 1), i]) ** 2)

                # PDE residuals for generalization
                index_i_param = (X_f[:, -1] == param_f[i_param]).nonzero(as_tuple=True)[0]
                f_i = f[index_i_param]
                loss_f += self.loss_f(f_i)

        # Total loss
        loss = loss_obs + loss_f * pde_weights

        return loss, loss_bc, loss_init, loss_data, loss_other, loss_f

    def test_pde(self, X_sup_test, u_sup_test, model_test):
        """
        Define the testing function.

        Args:
            X_sup_test (torch.Tensor): Testing points.
            u_sup_test (torch.Tensor): Reference solution on testing points.
            model_test (torch.nn.Module): Neural network model.

        Returns:
            torch.Tensor: Mean squared error ratio for testing data.
        """
        u_pred_test = self.net_transform(X_sup_test, model_test)
        mse_pred = torch.mean((u_pred_test - u_sup_test) ** 2)
        mse_ref = torch.mean(u_sup_test ** 2)
        return mse_pred / mse_ref

    def get_grad(self, X_f, param_f):
        """
        Calculate the gradients of the cost function w.r.t. training variables.

        Args:
            X_f (torch.Tensor): Collocation points.
            param_f (torch.Tensor): Parameter of the PDE.

        Returns:
            tuple: (loss_value, loss_bc, loss_init, loss_data, loss_other, loss_f, grads)
        """
        # Enable gradient tracking
        self.net_u.train()  # Ensure the model is in training mode
        self.optimizer.zero_grad()  # Clear previous gradients

        # Compute predictions for boundary conditions
        u_pred_bc = 0
        if self.nb_bc > 0:
            if self.type_problem == 'inverse':
                u_pred_bc = self.net_bc(self.X_bc, self.net_u, param_f)
            else:
                u_pred_bc = self.net_bc(self.X_bc, self.net_u)

        # Compute predictions for initial conditions
        u_pred_init = 0
        if self.nb_init > 0:
            if self.type_problem == 'inverse':
                u_pred_init = self.net_init(self.X_init, self.net_u, param_f)
            else:
                u_pred_init = self.net_init(self.X_init, self.net_u)

        # Compute predictions for data points
        u_pred_data = 0
        if self.nb_data > 0:
            u_pred_data = self.net_transform(self.X_data, self.net_u)

        # Compute predictions for other boundary conditions
        u_pred_other = 0
        if self.nb_other > 0:
            if self.type_problem == 'inverse':
                u_pred_other = self.net_other(self.X_other, self.net_u, param_f)
            else:
                u_pred_other = self.net_other(self.X_other, self.net_u)

        # Compute total loss and its components
        loss_value, loss_bc, loss_init, loss_data, loss_other, loss_f = self.loss_pinns(
            X_f, param_f, self.net_u, u_pred_bc, self.u_bc, u_pred_init, self.u_init,
            u_pred_data, self.u_data, u_pred_other, self.u_other, self.X_traction, self.pde_weights
        )

        # Compute gradients with respect to the trainable parameters
        loss_value.backward()  # Compute gradients
        grads = [param.grad.clone() if param.grad is not None else torch.zeros_like(param) for param in self.pinns_training_variables()]

        return loss_value, loss_bc, loss_init, loss_data, loss_other, loss_f, grads

    def train(self, max_epochs_adam=0, max_epochs_lbfgs=0, print_per_epochs=1000):
        """
        Train the neural networks.

        Args:
            max_epochs_adam (int): Maximum number of epochs for Adam optimizer.
            max_epochs_lbfgs (int): Maximum number of epochs for L-BFGS optimizer.
            print_per_epochs (int): Print the loss after a certain number of epochs.
        """

        def train_step(X_f, param_f):
            """
            Single training step using the Adam optimizer.
            """
            loss_value, loss_bc, loss_init, loss_data, loss_other, loss_f, grads = self.get_grad(X_f, param_f)

            # Apply gradients using optimizer
            self.optimizer.zero_grad()
            for param, grad in zip(self.pinns_training_variables(), grads):
                if grad is not None:
                    param.grad = grad
            self.optimizer.step()

            return loss_value, loss_bc, loss_init, loss_data, loss_other, loss_f

        # Adam Training
        for epoch in range(max_epochs_adam):
            loss_value, loss_bc, loss_init, loss_data, loss_other, loss_f = train_step(self.X_colloc, self.param_pde)

            if self.epoch % print_per_epochs == 0:
                print(f"Loss at epoch {self.epoch} (Adam): {loss_value.item()}")
            self.loss_array = np.append(self.loss_array, loss_value.item())
            if self.type_problem == 'inverse':
                self.param_pde_array = np.append(self.param_pde_array, self.param_pde.detach().numpy())

            # Evaluate on testing data if available
            if self.X_test is not None and epoch % 1000 == 0:
                if self.nb_param == 1:
                    res_test = self.test_pde(self.X_test, self.u_test, self.net_u)
                    self.test_array = np.append(self.test_array, res_test.item())
                    if res_test.item() < self.thres ** 2:
                        break
                else:
                    res_test_array = []
                    size_test = self.u_test.shape[0] // self.nb_param
                    for i_param in range(self.nb_param):
                        res_test = self.test_pde(
                            self.X_test[size_test * i_param:size_test * (i_param + 1)],
                            self.u_test[size_test * i_param:size_test * (i_param + 1)],
                            self.net_u
                        )
                        res_test_array.append(res_test.item())
                    if np.mean(res_test_array) < self.thres ** 2:
                        break

            self.epoch += 1

        # Callback for L-BFGS
        def callback(x):
            if self.type_problem == 'inverse':
                self.param_pde_array = np.append(self.param_pde_array, self.param_pde.detach().numpy())
            if self.epoch % print_per_epochs == 0:
                print(f"Loss at epoch {self.epoch} (L-BFGS): {self.current_loss}")
            self.epoch += 1

        # L-BFGS Optimizer
        def optimizer_lbfgs(X_f, param_f, method='L-BFGS-B', **kwargs):
            """
            L-BFGS optimizer to minimize the loss.
            """

            def get_weight():
                # Get current weights of the model
                weights = []
                for param in self.pinns_training_variables():
                    weights.extend(param.detach().numpy().flatten())
                return np.array(weights, dtype=np.float64)

            def set_weight(weights):
                # Update model weights with new values
                index = 0
                for param in self.pinns_training_variables():
                    shape = param.shape
                    size = np.prod(shape)
                    new_values = torch.tensor(weights[index:index + size], dtype=torch.float64).reshape(shape)
                    param.data = new_values
                    index += size

            def get_loss_and_grad(weights):
                # Compute loss and gradients
                set_weight(weights)
                loss_value, loss_bc, loss_init, loss_data, loss_other, loss_f, grads = self.get_grad(X_f, param_f)
                self.loss_array = np.append(self.loss_array, loss_value.item())
                self.current_loss = loss_value.item()

                # Flatten gradients
                grad_flat = []
                for grad in grads:
                    if grad is not None:
                        grad_flat.extend(grad.detach().numpy().flatten())
                grad_flat = np.array(grad_flat, dtype=np.float64)

                return loss_value.item(), grad_flat

            return scipy.optimize.minimize(
                fun=get_loss_and_grad,
                x0=get_weight(),
                jac=True,
                method=method,
                callback=callback,
                **kwargs
            )

        # Train using L-BFGS if specified
        if max_epochs_lbfgs > 0:
            if max_epochs_adam == 0:  # Initialize if no Adam training was done
                with torch.no_grad():
                    self.net_u(self.X_colloc)

            optimizer_lbfgs(
                self.X_colloc,
                self.param_pde,
                method='L-BFGS-B',
                options={
                    'maxiter': max_epochs_lbfgs,
                    'maxfun': max_epochs_lbfgs,
                    'maxcor': 100,
                    'maxls': 100,
                    'ftol': 0,
                    'gtol': 1.0 * np.finfo(float).eps
                }
            )