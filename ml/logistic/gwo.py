import numpy as np

# --- MULTICLASS GWO (With Hyperparameter C) ---
def gwo_multiclass(X_train, y_train, X_valid, y_valid, fitness_function, target_n_features=None, n_wolves=20, n_iterations=500):
    n_features = X_train.shape[1]
    n_attributes = n_features + 1
    positions = np.random.uniform(0, 1, size=(n_wolves, n_attributes))
    fitness_history = []

    alpha_position = np.zeros(n_attributes)
    beta_position = np.zeros(n_attributes)
    delta_position = np.zeros(n_attributes)

    alpha_score = -np.inf
    beta_score = -np.inf
    delta_score = -np.inf

    for iteration in range(n_iterations):
        for i in range(n_wolves):
            binary_wolf = np.zeros(n_features)
            feature_positions = positions[i, :-1]
            if target_n_features is not None and 1 <= target_n_features <= n_features:
                indices = np.argsort(feature_positions)[-target_n_features:]
                binary_wolf[indices] = 1
            else:
                binary_wolf = (feature_positions >= 0.5).astype(int)
            
            c_val = positions[i, -1] * (10.0 - 0.001) + 0.001
            score = fitness_function(binary_wolf, X_train, y_train, X_valid, y_valid, c_val=c_val)

            if score > alpha_score:
                delta_score, delta_position = beta_score, beta_position.copy()
                beta_score, beta_position = alpha_score, alpha_position.copy()
                alpha_score, alpha_position = score, positions[i].copy()
            elif score > beta_score:
                delta_score, delta_position = beta_score, beta_position.copy()
                beta_score, beta_position = score, positions[i].copy()
            elif score > delta_score:
                delta_score, delta_position = score, positions[i].copy()

        a = 2 - iteration * (2 / n_iterations)

        for i in range(n_wolves):
            for j in range(n_attributes):
                r1, r2 = np.random.rand(), np.random.rand()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha_position[j] - positions[i, j])
                X1 = alpha_position[j] - A1 * D_alpha

                r1, r2 = np.random.rand(), np.random.rand()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta_position[j] - positions[i, j])
                X2 = beta_position[j] - A2 * D_beta

                r1, r2 = np.random.rand(), np.random.rand()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta_position[j] - positions[i, j])
                X3 = delta_position[j] - A3 * D_delta

                new_position = (X1 + X2 + X3) / 3.0
                transfer = 1 / (1 + np.exp(-new_position))
                
                if j < n_features:
                    positions[i, j] = 1.0 if np.random.rand() < transfer else 0.0
                else:
                    positions[i, j] = np.clip(new_position, 0.0, 1.0)

        fitness_history.append(alpha_score)

    best_wolf = np.zeros(n_features)
    alpha_feature_positions = alpha_position[:-1]
    if target_n_features is not None and 1 <= target_n_features <= n_features:
        indices = np.argsort(alpha_feature_positions)[-target_n_features:]
        best_wolf[indices] = 1
    else:
        best_wolf = (alpha_feature_positions >= 0.5).astype(int)
        
    best_c_val = alpha_position[-1] * (10.0 - 0.001) + 0.001
        
    return best_wolf, best_c_val, alpha_score, fitness_history

# --- BINARY LEGACY GWO ---
def gwo_binary(X_train, y_train, X_valid, y_valid, fitness_function, target_n_features=None, n_wolves=20, n_iterations=500):
    n_features = X_train.shape[1]
    positions = np.random.uniform(0, 1, size=(n_wolves, n_features))
    fitness_history = []

    alpha_position = np.zeros(n_features)
    beta_position = np.zeros(n_features)
    delta_position = np.zeros(n_features)

    alpha_score = -np.inf
    beta_score = -np.inf
    delta_score = -np.inf

    for iteration in range(n_iterations):
        for i in range(n_wolves):
            binary_wolf = np.zeros(n_features)
            if target_n_features is not None and 1 <= target_n_features <= n_features:
                indices = np.argsort(positions[i])[-target_n_features:]
                binary_wolf[indices] = 1
            else:
                binary_wolf = (positions[i] >= 0.5).astype(int)
            
            score = fitness_function(binary_wolf, X_train, y_train, X_valid, y_valid)

            if score > alpha_score:
                delta_score, delta_position = beta_score, beta_position.copy()
                beta_score, beta_position = alpha_score, alpha_position.copy()
                alpha_score, alpha_position = score, positions[i].copy()
            elif score > beta_score:
                delta_score, delta_position = beta_score, beta_position.copy()
                beta_score, beta_position = score, positions[i].copy()
            elif score > delta_score:
                delta_score, delta_position = score, positions[i].copy()

        a = 2 - iteration * (2 / n_iterations)

        for i in range(n_wolves):
            for j in range(n_features):
                r1, r2 = np.random.rand(), np.random.rand()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha_position[j] - positions[i, j])
                X1 = alpha_position[j] - A1 * D_alpha

                r1, r2 = np.random.rand(), np.random.rand()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta_position[j] - positions[i, j])
                X2 = beta_position[j] - A2 * D_beta

                r1, r2 = np.random.rand(), np.random.rand()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta_position[j] - positions[i, j])
                X3 = delta_position[j] - A3 * D_delta

                new_position = (X1 + X2 + X3) / 3.0
                transfer = 1 / (1 + np.exp(-new_position))
                positions[i, j] = 1.0 if np.random.rand() < transfer else 0.0

        fitness_history.append(alpha_score)

    best_wolf = np.zeros(n_features)
    if target_n_features is not None and 1 <= target_n_features <= n_features:
        indices = np.argsort(alpha_position)[-target_n_features:]
        best_wolf[indices] = 1
    else:
        best_wolf = (alpha_position >= 0.5).astype(int)
        
    return best_wolf, alpha_score, fitness_history
