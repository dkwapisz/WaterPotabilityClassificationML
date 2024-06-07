import pickle
import random
import time

import numpy as np
import optuna
from deap import base, creator, tools, algorithms
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV

with open('data_dump/normalizedStdInterpolateVars.pkl', 'rb') as f:
    normalized_std_interpolate = pickle.load(f)
    scaler_std_interpolate = pickle.load(f)


random_seed = 42
random.seed(random_seed)

def split_df_train_test(data, test_size, seed):
    np.random.seed(seed)

    unique_labels = data['Potability'].unique()
    label_counts = data['Potability'].value_counts()

    test_indices = []

    for label in unique_labels:
        num_label_samples = label_counts[label]
        num_test_samples = int(test_size * num_label_samples)
        label_indices = data.index[data['Potability'] == label].tolist()
        label_test_indices = np.random.choice(label_indices, size=num_test_samples, replace=False)
        test_indices.extend(label_test_indices)

    train_indices = np.setdiff1d(data.index, test_indices)

    train_set = data.loc[train_indices]
    test_set = data.loc[test_indices]

    return train_set, test_set


def calculate_f1_score(y_true, y_pred):
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    true_negatives = np.sum((y_true == 0) & (y_pred == 0))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))

    if (true_positives + false_positives) == 0:
        print("stop")

    print(f"TP: {true_positives}, TN: {true_negatives}, FP: {false_positives}, FN: {false_negatives}")

    precision_positives = true_positives / (true_positives + false_positives)
    recall_positives = true_positives / (true_positives + false_negatives)
    f1_score_positives = 2 * (precision_positives * recall_positives) / (precision_positives + recall_positives)

    precision_negatives = true_negatives / (true_negatives + false_negatives)
    recall_negatives = true_negatives / (true_negatives + false_positives)
    f1_score_negatives = 2 * (precision_negatives * recall_negatives) / (precision_negatives + recall_negatives)

    f1_score = (f1_score_positives + f1_score_negatives) / 2
    return f1_score


train_std_interpolate, test_std_interpolate = split_df_train_test(normalized_std_interpolate, 0.2, 123)

train_class_counts = train_std_interpolate['Potability'].value_counts(normalize=True) * 100
test_class_counts = test_std_interpolate['Potability'].value_counts(normalize=True) * 100

print("Train set:")
print(train_class_counts)
print("\nTest set:")
print(test_class_counts)

class RandomForestClassifierWrapper:

    def __init__(self, train_set, test_set, optimize_f1_score):
        self.classifier_name = "Random Forest"
        self.train_set = train_set.iloc[:, :-1]
        self.test_set = test_set.iloc[:, :-1]
        self.train_label = train_set.iloc[:, -1]
        self.test_label = test_set.iloc[:, -1]
        self.model = None
        self.test_pred = None
        self.hof = None
        self.best_optuna_trial = None
        self.best_score = None
        self.best_score_type = "F1 score" if optimize_f1_score else "accuracy"
        self.best_params = None
        self.processing_time = None
        self._create_model()

    # ----------------- DEAP -----------------
    def deap_objective(self, individual):
        n_estimators, max_depth, min_samples_split, min_samples_leaf, bootstrap, criterion = individual
        n_estimators = int(n_estimators)
        max_depth = int(max_depth) if max_depth > 0 else None
        min_samples_split = int(min_samples_split) if int(min_samples_split) > 1 else 2
        min_samples_leaf = int(min_samples_leaf) if int(min_samples_leaf) > 0 else 1
        bootstrap = bool(bootstrap)
        criterion = 'gini' if criterion < 0.5 else 'entropy'
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                       min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                       bootstrap=bootstrap, criterion=criterion, random_state=random_seed)
        self.model.fit(self.train_set, self.train_label)
        predictions = self.model.predict(self.test_set)

        if self.best_score_type == "F1 score":
            score = calculate_f1_score(self.test_label, predictions)
        else:
            score = accuracy_score(self.test_label, predictions)

        return score,

    def train_model_with_ga(self):
        if not hasattr(creator, 'FitnessMax'):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, 'Individual'):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        toolbox.register("attr_n_estimators", random.randint, 10, 200)
        toolbox.register("attr_max_depth", random.randint, 1, 50)
        toolbox.register("attr_min_samples_split", random.randint, 2, 10)
        toolbox.register("attr_min_samples_leaf", random.randint, 1, 10)
        toolbox.register("attr_bootstrap", random.randint, 0, 1)
        toolbox.register("attr_criterion", random.uniform, 0, 1)

        toolbox.register("individual", tools.initCycle, creator.Individual,
                         (toolbox.attr_n_estimators, toolbox.attr_max_depth, toolbox.attr_min_samples_split,
                          toolbox.attr_min_samples_leaf, toolbox.attr_bootstrap, toolbox.attr_criterion), n=1)

        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", self.deap_objective)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=5)

        pop = toolbox.population(n=100)

        self.hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=30, lambda_=50, cxpb=0.5, mutpb=0.2,
                                                 ngen=5,
                                                 stats=stats, halloffame=self.hof, verbose=True)

        best_individual = tools.selBest(pop, 1)[0]

        self.model = RandomForestClassifier(n_estimators=int(best_individual[0]),
                                            max_depth=int(best_individual[1]),
                                            min_samples_split=int(
                                                best_individual[2] if int(best_individual[2]) > 1 else 2),
                                            min_samples_leaf=int(
                                                best_individual[3] if int(best_individual[3]) > 0 else 1),
                                            bootstrap=bool(best_individual[4]),
                                            criterion='gini' if best_individual[5] < 0.5 else 'entropy',
                                            random_state=random_seed)
        self.model.fit(self.train_set, self.train_label)
        self.set_best_score()
        self.save_best_params_deap()

    def save_best_params_deap(self):
        self.best_params = {
            "n_estimators": int(self.hof[0][0]),
            "max_depth": int(self.hof[0][1]),
            "min_samples_split": int(self.hof[0][2]),
            "min_samples_leaf": int(self.hof[0][3]),
            "bootstrap": bool(self.hof[0][4]),
            "criterion": 'gini' if self.hof[0][5] < 0.5 else 'entropy'
        }

    # ----------------- Optuna -----------------
    def optuna_objective(self, trial):
        n_estimators = trial.suggest_int("n_estimators", 10, 200)
        max_depth = trial.suggest_int("max_depth", 1, 50)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
        bootstrap = trial.suggest_categorical("bootstrap", [True, False])
        criterion = trial.suggest_categorical("criterion", ['gini', 'entropy'])

        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                       min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                       bootstrap=bootstrap, criterion=criterion, random_state=random_seed)
        model.fit(self.train_set, self.train_label)
        predictions = model.predict(self.test_set)

        if self.best_score_type == "F1 score":
            score = calculate_f1_score(self.test_label, predictions)
        else:
            score = accuracy_score(self.test_label, predictions)

        return score

    def train_model_with_optuna(self, n_trials=100):
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.optuna_objective(trial), n_jobs=-1, n_trials=n_trials)

        self.best_optuna_trial = study.best_trial

        self.model = RandomForestClassifier(n_estimators=self.best_optuna_trial.params["n_estimators"],
                                            max_depth=self.best_optuna_trial.params["max_depth"],
                                            min_samples_split=self.best_optuna_trial.params["min_samples_split"],
                                            min_samples_leaf=self.best_optuna_trial.params["min_samples_leaf"],
                                            bootstrap=self.best_optuna_trial.params["bootstrap"],
                                            criterion=self.best_optuna_trial.params["criterion"],
                                            random_state=random_seed)
        self.model.fit(self.train_set, self.train_label)
        self.set_best_score()
        self.save_best_params_optuna()

    def save_best_params_optuna(self):
        self.best_params = {
            "n_estimators": int(self.best_optuna_trial.params["n_estimators"]),
            "max_depth": int(self.best_optuna_trial.params["max_depth"]),
            "min_samples_split": int(self.best_optuna_trial.params["min_samples_split"]),
            "min_samples_leaf": int(self.best_optuna_trial.params["min_samples_leaf"]),
            "bootstrap": bool(self.best_optuna_trial.params["bootstrap"]),
            "criterion": self.best_optuna_trial.params["criterion"]
        }

    # ----------------- RandomizedSearchCV -----------------
    def train_model_CV(self):
        param_distributions = {
            "n_estimators": list(range(10, 201)),
            "max_depth": [None] + list(range(1, 51)),
            "min_samples_split": list(range(2, 11)),
            "min_samples_leaf": list(range(1, 11)),
            "bootstrap": [True, False],
            "criterion": ['gini', 'entropy']
        }

        if self.best_score_type == "F1 score":
            scoring = 'f1'
        else:
            scoring = 'accuracy'

        random_search = RandomizedSearchCV(self.model, param_distributions=param_distributions, scoring=scoring,
                                           n_iter=12, cv=5, random_state=random_seed, n_jobs=-1)
        random_search.fit(self.train_set, self.train_label)
        self.model = random_search.best_estimator_

    def evaluate_model_CV(self):
        self.set_best_score()
        self.save_best_params_CV()

    def set_best_score(self):
        self.test_pred = self.model.predict(self.test_set)
        if self.best_score_type == "F1 score":
            self.best_score = calculate_f1_score(self.test_label, self.test_pred)
        else:
            self.best_score = accuracy_score(self.test_label, self.test_pred)

    def save_best_params_CV(self):
        self.best_params = {
            "n_estimators": int(self.model.get_params()['n_estimators']),
            "max_depth": int(self.model.get_params()['max_depth']),
            "min_samples_split": int(self.model.get_params()['min_samples_split']),
            "min_samples_leaf": int(self.model.get_params()['min_samples_leaf']),
            "bootstrap": bool(self.model.get_params()['bootstrap']),
            "criterion": self.model.get_params()['criterion']
        }

    # ----------------- Other -----------------
    def _create_model(self):
        self.model = RandomForestClassifier(random_state=random_seed)

    def set_processing_time(self, end, start):
        self.processing_time = end - start


if __name__ == "__main__":
    start_time = time.time()
    RF_model_f1score_optimizing_DEAP = RandomForestClassifierWrapper(train_std_interpolate, test_std_interpolate,
                                                                     optimize_f1_score=True)
    RF_model_f1score_optimizing_DEAP.train_model_with_ga()
    end_time = time.time()
    RF_model_f1score_optimizing_DEAP.set_processing_time(end_time, start_time)