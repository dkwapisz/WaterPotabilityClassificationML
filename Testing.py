# Std zeros
modelMLP_std = MLPClassifierWrapper(train_std_0, test_std_0)
modelMLP_std.train_model()
accuracy_MLP_std_zeros = modelMLP_std.evaluate_model()

# Std mean
modelMLP_std = MLPClassifierWrapper(train_std_mean, test_std_mean)
modelMLP_std.train_model()
accuracy_MLP_std_mean = modelMLP_std.evaluate_model()

# Std interpolate
modelMLP_std = MLPClassifierWrapper(train_std_interpolate, test_std_interpolate)
modelMLP_std.train_model()
accuracy_MLP_std_interpolate = modelMLP_std.evaluate_model()

# Minmax zeros
modelMLP_minmax = MLPClassifierWrapper(train_minmax_0, test_minmax_0)
modelMLP_minmax.train_model()
accuracy_MLP_minmax_0 = modelMLP_minmax.evaluate_model()

# Minmax mean
modelMLP_minmax = MLPClassifierWrapper(train_minmax_mean, test_minmax_mean)
modelMLP_minmax.train_model()
accuracy_MLP_minmax_mean = modelMLP_minmax.evaluate_model()

# Minmax interpolate
modelMLP_minmax = MLPClassifierWrapper(train_minmax_interpolate, test_minmax_interpolate)
modelMLP_minmax.train_model()
accuracy_MLP_minmax_interpolate = modelMLP_minmax.evaluate_model()