import numpy as np
from scipy.stats import loguniform
from .nnf_helper import split_module_name, dynamic_import, config_to_str
from .dj_helpers import make_hash
# from nnfabrik.main import * # avoid nnfabrik_core
import datajoint as dj
import optuna
from optuna.trial import TrialState
import gc

class nnfabrikOptuna:
    """
    A hyperparameter optimization tool based on Optuna (https://optuna.org/) integrated with nnfabrik.
    This tool, iteratively, optimizes for hyperparameters that improve a specific score representing
    model performance (the same score used in TrainedModel table). In every iteration (after every training),
    it automatically adds an entry to the corresponding tables, and populated the trained model table (i.e.
    trains the model) for that specific entry.

    Args:
        optuna_study_config (dict): dictionary of arguments for optuna study: 
            storage: str | storages.BaseStorage | None = None,
            sampler: "samplers.BaseSampler" | None = None,
            pruner: pruners.BasePruner | None = None,
            study_name: str | None = None,
            direction: str | StudyDirection | None = None,
            load_if_exists: bool = False,
            directions: Sequence[str | StudyDirection] | None = None,

        optuna_optimization_config (dict): dictionary of arguments for optuna optimization:
            func: ObjectiveFuncType,
            n_trials: int | None = None,
            timeout: float | None = None,
            n_jobs: int = 1,
            catch: Iterable[type[Exception]] | type[Exception] = (),
            callbacks: Iterable[Callable[[Study, FrozenTrial], None]] | None = None,
            gc_after_trial: bool = False,
            show_progress_bar: bool = False,
        
        dataset_fn (str): name of the dataset function
        dataset_config (dict): dictionary of arguments for dataset function that are fixed
        dataset_config_auto (dict): dictionary of arguments for dataset function that are to be optimized
        set it to {} if no arguments are to be optimized

        model_fn (str): name of the model function
        model_config (dict): dictionary of arguments for model function that are fixed
        model_config_auto (dict): dictionary of arguments for model function that are to be optimized
            example format:
            model_config_auto = {
                    "hidden_channels": trial.suggest_int("hidden_channels", 2, 4),
                    "GNN_model": trial.suggest_categorical("GNN_model", ['GCNConv', 'GraphConv', 'ChebConv',
                     'TAGConv', 'SGConv', 'SSGConv', 'MFConv']),
                    "ChebConv_k": trial.suggest_int("k", 1, 5),
                    "SSGConv_k": trial.suggest_int("k", 0, 1),
                }

        trainer_fn (str): name of the trainer function
        trainer_config (dict): dictionary of arguments for trainer function that are fixed
        trainer_config_auto (dict): dictionary of arguments for trainer function that are to be optimized
            example format:
            trainer_config_auto = {
                    "batch_size": trial.suggest_int("batch_size", 16, 128),
                    "lr": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
                    "weight_decay": trial.suggest_float("weight_decay", 0.00001,  0.0001, log = True),
                    "optimizer": trial.suggest_categorical("optimizer", ["adam", "sgd"]),
                }

        architect (str): Name of the contributor that added this entry
        nn_module: python classes for current schema. this would have necessary make functions for doing autopoluate
        comment (str, optional): Comments about this optimization round. It will be used to fill up the comment entry of dataset, model, and trainer table. Defaults to "Bayesian optimization of Hyper params.".
    """

    def __init__(
        self,
        optuna_study_config, # optuna study related configurations
        optuna_optimization_config, # optuna optimization related configurations
        dataset_fn,
        dataset_config, # dictionary of fixed arguments for dataset function
        dataset_config_tune, # dictionary of arguments for dataset function that are to be optimized
        model_fn,
        model_config,
        model_config_tune,
        trainer_fn,
        trainer_config,
        trainer_config_tune,
        architect,
        nn_module,
        comment="Optimization of Hyper params using Optuna in nnfabrik.",
        
    ):
        self.optuna_study_config = optuna_study_config ## TODO: save those info in trained_model table?
        self.optuna_optimization_config = optuna_optimization_config ## TODO: save those info in trained_model table?
        self.fns = dict(dataset=dataset_fn, model=model_fn, trainer=trainer_fn)
        self.dataset_config = dataset_config
        self.model_config = model_config
        self.trainer_config = trainer_config
        self.dataset_config_tune = dataset_config_tune
        self.model_config_tune = model_config_tune
        self.trainer_config_tune = trainer_config_tune
        self.architect = architect
        self.comment = comment
        self.data_set_table = nn_module.Dataset()
        self.trainer_table = nn_module.Trainer()
        self.model_table = nn_module.Model()
        self.trained_model_table = nn_module.TrainedModel

    def optuna_params_eval(self, trial: optuna.Trial, param_specs: dict) -> dict:
        params = {}
        for key, spec in param_specs.items():
            condition = spec.get("condition", lambda p: True)
            if not condition(params):
                continue  # skip conditional param
            method = spec["method"]
            args = spec["args"]
            default = spec.get("default")

            try:
                suggest_func = getattr(trial, method)
                params[key] = suggest_func(**args)
            except Exception as e:
                if default is not None:
                    params[key] = default
                else:
                    raise RuntimeError(f"Failed to evaluate {key}: {e}")
        return params


    def make_new_config(self, trial: optuna.Trial, fix_config: dict, tune_config: dict) -> (dict, str):
        """
        For each trial, generate  a new configuration by combining the fixed and tunable configurations.
        Args:
            trial: optuna trial object
            fix_config: fixed configuration
            tune_config: tunable configuration
        Returns:
            config: configuration
            config_hash: hash of the configuration
        """
        
        config = fix_config.copy()
        config_tune = self.optuna_params_eval(trial, tune_config)
        config.update(config_tune)
        config_hash = make_hash(config)
        config_tune_str = config_to_str(config_tune)
        return (config, config_hash, config_tune_str)

    def insert_new_entry(self, trial: optuna.Trial, table_name: str,  _config: dict, _config_tune: dict):
        ''' helper function to insert a new entry to one of the tables with giving configs
        return:
            hash for the primary key
        '''
        assert table_name in ['dataset', 'model', 'trainer'], 'invalid table name'
        table_dict = dict(dataset=self.data_set_table, model=self.model_table,trainer=self.trainer_table)
        dj_table = table_dict[table_name]
        config_, hash_, cmt_ = self.make_new_config(trial, _config, _config_tune)
        query_dict = {f'{table_name}_fn':self.fns[table_name], f'{table_name}_hash': hash_}      
        if not (query_dict in dj_table):
            insert_dict = {**query_dict, f'{table_name}_config':config_, \
                           f'{table_name}_fabrikant': self.architect, f'{table_name}_comment': cmt_}
            dj_table.insert1(insert_dict, skip_duplicates=True) # add_entry
        return hash_
        ...
    def make_nnfabric_spec(self, trial: optuna.Trial):
        """
        For a given set of parameters, add an entry to the corresponding tables, and populate the trained model
        table for that specific entry.

        Args:
            trial:

        Returns:
            dict:
        """
        dataset_hash = self.insert_new_entry(trial, 'dataset', self.dataset_config, self.dataset_config_tune)
        model_hash = self.insert_new_entry(trial, 'model', self.model_config, self.model_config_tune)
        trainer_hash = self.insert_new_entry(trial, 'trainer', self.trainer_config, self.trainer_config_tune)
        # get the primary key values for all those entries
        restriction = (
            'dataset_fn in ("{}")'.format(self.fns["dataset"]),
            'dataset_hash in ("{}")'.format(dataset_hash),
            'model_fn in ("{}")'.format(self.fns["model"]),
            'model_hash in ("{}")'.format(model_hash),
            'trainer_fn in ("{}")'.format(self.fns["trainer"]),
            'trainer_hash in ("{}")'.format(trainer_hash),
        )
        trial.set_user_attr("trial_restriction", restriction)
        return restriction

    def objective(self, trial: optuna.Trial) -> float | tuple:
        """
        Objective function to be optimized by Optuna.
        returns:
            Metric(s) to be optimized.
        """
        nnfabric_spec = self.make_nnfabric_spec(trial)
        # populate the table for those primary keys
        self.trained_model_table.populate(*nnfabric_spec)
        # fetch scores
        scores = (self.trained_model_table & dj.AndList(nnfabric_spec)).fetch("scores")
        if len(scores) == 1:
            return scores[0]
        elif len(scores) == 2:
            return tuple(scores)
        else:
            raise ValueError("Unexpected number of scores fetched.")

    def populate_optuna_trial_info(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        ''' update trial information with call back'''
        ...
        tuna_trial_info = dict(optuna_trial_number = trial.number, optuna_trial_state = str(trial.state))
        nnfabric_spec = trial.user_attrs["trial_restriction"]  
        key = (self.trained_model_table & dj.AndList(nnfabric_spec)).fetch1("KEY")         
        update_data = {**key, **tuna_trial_info}
        self.trained_model_table.update1(update_data)
        del update_data
        gc.collect()
        print('------------ Trial finished and GC performed!-----------')

    def run(self, report=True):
        """
        Runs an Optuna study
        returns:
            study: optuna.study.Study: The study object containing the optimization results.
        """
        study = optuna.create_study(**self.optuna_study_config)
        study.optimize(self.objective, **self.optuna_optimization_config, callbacks=[self.populate_optuna_trial_info])
        self.study = study
        if report:
            pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
            complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
            print("Study statistics: ")
            print("  Number of finished trials: ", len(study.trials))
            print("  Number of pruned trials: ", len(pruned_trials))
            print("  Number of complete trials: ", len(complete_trials))
        return study

class Bayesian:
    """
    A hyperparameter optimization tool based on Facebook Ax (https://ax.dev/), integrated with nnfabrik.
    This tool, iteratively, optimizes for hyperparameters that improve a specific score representing
    model performance (the same score used in TrainedModel table). In every iteration (after every training),
    it automatically adds an entry to the corresponding tables, and populated the trained model table (i.e.
    trains the model) for that specific entry.

    Args:
        dataset_fn (str): name of the dataset function
        dataset_config (dict): dictionary of arguments for dataset function that are fixed
        dataset_config_auto (dict): dictionary of arguments for dataset function that are to be optimized
        model_fn (str): name of the model function
        model_config (dict): dictionary of arguments for model function that are fixed
        model_config_auto (dict): dictionary of arguments for model function that are to be optimized
        trainer_fn (str): name of the trainer function
        trainer_config (dict): dictionary of arguments for trainer function that are fixed
        trainer_config_auto (dict): dictionary of arguments for trainer function that are to be optimized
        architect (str): Name of the contributor that added this entry
        trained_model_table (str): name (importable) of the trained_model_table
        total_trials (int, optional): Number of experiments (i.e. training) to run. Defaults to 5.
        arms_per_trial (int, optional): Number of different configurations used for training (for more details check https://ax.dev/docs/glossary.html#trial). Defaults to 1.
        comment (str, optional): Comments about this optimization round. It will be used to fill up the comment entry of dataset, model, and trainer table. Defaults to "Bayesian optimization of Hyper params.".
    """

    def __init__(
        self,
        dataset_fn,
        dataset_config,
        dataset_config_auto,
        model_fn,
        model_config,
        model_config_auto,
        trainer_fn,
        trainer_config,
        trainer_config_auto,
        architect,
        trained_model_table,
        total_trials=5,
        arms_per_trial=1,
        comment="Bayesian optimization of Hyper params.",
    ):

        self.fns = dict(dataset=dataset_fn, model=model_fn, trainer=trainer_fn)
        self.fixed_params = self.get_fixed_params(dataset_config, model_config, trainer_config)
        self.auto_params = self.get_auto_params(dataset_config_auto, model_config_auto, trainer_config_auto)
        self.architect = architect
        self.total_trials = total_trials
        self.arms_per_trial = arms_per_trial
        self.comment = comment

        # import TrainedModel definition
        module_path, class_name = split_module_name(trained_model_table)
        self.trained_model_table = dynamic_import(module_path, class_name)

    @staticmethod
    def get_fixed_params(dataset_config, model_config, trainer_config):
        """
        Returns a single dictionary including the fixed parameters for dataset, model, and trainer.

        Args:
            dataset_config (dict): dictionary of arguments for dataset function that are fixed
            model_config (dict): dictionary of arguments for model function that are fixed
            trainer_config (dict): dictionary of arguments for trainer function that are fixed

        Returns:
            dict: A dictionary of dictionaries where keys are dataset, model, and trainer and the values
            are the corresponding dictionary of fixed arguments.
        """
        return dict(dataset=dataset_config, model=model_config, trainer=trainer_config)

    @staticmethod
    def get_auto_params(dataset_config_auto, model_config_auto, trainer_config_auto):
        """
        Changes the parameters to be optimized to a ax-friendly format, i.e. a list of dictionaries where
        each entry of the list corresponds to a single parameter.

        Ax requires a list of parameters (to be optimized) including the name and other specifications.
        Here we provide that list while keeping the arguments still separated by adding "dataset",
        "model", or "trainer" to the beginning of the parameter name.

        Args:
            dataset_config_auto (dict): dictionary of arguments for dataset function that are to be optimized
            model_config_auto (dict): dictionary of arguments for model function that are to be optimized
            trainer_config_auto (dict): dictionary of arguments for trainer function that are to be optimized

        Returns:
            list: list of dictionaries where each dictionary specifies a single parameter to be optimized.
        """
        dataset_params = []
        for k, v in dataset_config_auto.items():
            dd = {"name": "dataset.{}".format(k)}
            dd.update(v)
            dataset_params.append(dd)

        model_params = []
        for k, v in model_config_auto.items():
            dd = {"name": "model.{}".format(k)}
            dd.update(v)
            model_params.append(dd)

        trainer_params = []
        for k, v in trainer_config_auto.items():
            dd = {"name": "trainer.{}".format(k)}
            dd.update(v)
            trainer_params.append(dd)

        return dataset_params + model_params + trainer_params

    @staticmethod
    def _combine_params(auto_params, fixed_params):
        """
        Combining the auto (to-be-optimized) and fixed parameters (to have a single object representing all the arguments
        used for a specific function)

        Args:
            auto_params (dict): dictionary of to-be-optimized parameters, i.e. A dictionary of dictionaries where keys are
            dataset, model, and trainer and the values are the corresponding dictionary of to-be-optimized arguments.
            fixed_params (dict): dictionary of fixed parameters, i.e. A dictionary of dictionaries where keys are dataset,
            model, and trainer and the values are the corresponding dictionary of fixed arguments.

        Returns:
            dict: dictionary of parameters (fixed and to-be-optimized), i.e. A dictionary of dictionaries where keys are
            dataset, model, and trainer and the values are the corresponding dictionary of arguments.
        """
        keys = ["dataset", "model", "trainer"]
        params = {}
        for key in keys:
            params[key] = fixed_params[key]
            params[key].update(auto_params[key])

        return {key: params[key] for key in keys}

    @staticmethod
    def _split_config(params):
        """
        Reverses the operation of `get_auto_params` (from ax-friendly format to a dictionary of dictionaries where keys are
        dataset, model, and trainer and the values are a dictionary of the corresponding arguments)

        Args:
            params (dict): dictionary of dictionaries where each dictionary specifies a single parameter to be optimized.

        Returns:
            dict: A dictionary of dictionaries where keys are dataset, model, and trainer and the values are the
            corresponding dictionary of to-be-optimized arguments.
        """
        config = dict(dataset={}, model={}, trainer={}, others={})
        for k, v in params.items():
            config[k.split(".")[0]][k.split(".")[1]] = v

        return config

    def train_evaluate(self, auto_params):
        """
        For a given set of parameters, add an entry to the corresponding tables, and populate the trained model
        table for that specific entry.

        Args:
            auto_params (dict): dictionary of dictionaries where each dictionary specifies a single parameter to be optimized.

        Returns:
            float: the score of the trained model for the specific entry in trained model table
        """
        config = self._combine_params(self._split_config(auto_params), self.fixed_params)

        # insert the stuff into their corresponding tables
        dataset_hash = make_hash(config["dataset"])
        entry_exists = {
            "dataset_fn": "{}".format(self.fns["dataset"])
        } in self.trained_model_table().dataset_table() and {
            "dataset_hash": "{}".format(dataset_hash)
        } in self.trained_model_table().dataset_table()
        if not entry_exists:
            self.trained_model_table().dataset_table().add_entry(
                self.fns["dataset"],
                config["dataset"],
                dataset_fabrikant=self.architect,
                dataset_comment=self.comment,
            )

        model_hash = make_hash(config["model"])
        entry_exists = {"model_fn": "{}".format(self.fns["model"])} in self.trained_model_table().model_table() and {
            "model_hash": "{}".format(model_hash)
        } in self.trained_model_table().model_table()
        if not entry_exists:
            self.trained_model_table().model_table().add_entry(
                self.fns["model"],
                config["model"],
                model_fabrikant=self.architect,
                model_comment=self.comment,
            )

        trainer_hash = make_hash(config["trainer"])
        entry_exists = {
            "trainer_fn": "{}".format(self.fns["trainer"])
        } in self.trained_model_table().trainer_table() and {
            "trainer_hash": "{}".format(trainer_hash)
        } in self.trained_model_table().trainer_table()
        if not entry_exists:
            self.trained_model_table().trainer_table().add_entry(
                self.fns["trainer"],
                config["trainer"],
                trainer_fabrikant=self.architect,
                trainer_comment=self.comment,
            )

        # get the primary key values for all those entries
        restriction = (
            'dataset_fn in ("{}")'.format(self.fns["dataset"]),
            'dataset_hash in ("{}")'.format(dataset_hash),
            'model_fn in ("{}")'.format(self.fns["model"]),
            'model_hash in ("{}")'.format(model_hash),
            'trainer_fn in ("{}")'.format(self.fns["trainer"]),
            'trainer_hash in ("{}")'.format(trainer_hash),
        )

        # populate the table for those primary keys
        self.trained_model_table().populate(*restriction)

        # get the score of the model for this specific set of hyperparameters
        score = (self.trained_model_table() & dj.AndList(restriction)).fetch("score")[0]

        return score

    def run(self):
        """
        Runs Bayesian optimization.

        Returns:
            tuple: The returned values are similar to that of Ax (refer to https://ax.dev/docs/api.html)
        """
        best_parameters, values, experiment, model = optimize(
            parameters=self.auto_params,
            evaluation_function=self.train_evaluate,
            objective_name="val_corr",
            minimize=False,
            total_trials=self.total_trials,
            arms_per_trial=self.arms_per_trial,
        )

        return self._split_config(best_parameters), values, experiment, model


class Random:
    """
    Random hyperparameter search, integrated with nnfabrik.
    Similar to Bayesian optimization tool, but instead of optimizing for hyperparameters to maximize a score,
    in every iteration (after every training), it randomly samples new value for the specified parameters, adds an
    entry to the corresponding tables, and populated the trained model table (i.e. trains the model) for that specific entry.

    Args:
        dataset_fn (str): name of the dataset function
        dataset_config (dict): dictionary of arguments for dataset function that are fixed
        dataset_config_auto (dict): dictionary of arguments for dataset function that are to be randomly sampled
        model_fn (str): name of the model function
        model_config (dict): dictionary of arguments for model function that are fixed
        model_config_auto (dict): dictionary of arguments for model function that are to be randomly sampled
        trainer_fn (str): name of the trainer function
        trainer_config (dict): dictionary of arguments for trainer function that are fixed
        trainer_config_auto (dict): dictionary of arguments for trainer function that are to be randomly sampled
        seed_config_auto (dict): dictionary of arguments for setting (`dict(seed={"type": "fixed", "value": <VALUE>})`)
            or random sampling (`dict(seed={"type": "int"})`) the seed
        architect (str): Name of the contributor that added this entry
        trained_model_table (str): name (importable) of the trained_model_table
        total_trials (int, optional): Number of experiments (i.e. training) to run. Defaults to 5.
        comment (str, optional): Comments about this optimization round. It will be used to fill up the comment entry of dataset, model, and trainer table. Defaults to "Bayesian optimization of Hyper params.".
    """

    def __init__(
        self,
        dataset_fn,
        dataset_config,
        dataset_config_auto,
        model_fn,
        model_config,
        model_config_auto,
        trainer_fn,
        trainer_config,
        trainer_config_auto,
        seed_config_auto,
        architect,
        trained_model_table,
        total_trials=5,
        comment="Random search for hyper params.",
    ):

        self.fns = dict(dataset=dataset_fn, model=model_fn, trainer=trainer_fn)
        self.fixed_params = self.get_fixed_params(dataset_config, model_config, trainer_config)
        self.auto_params = self.get_auto_params(
            dataset_config_auto, model_config_auto, trainer_config_auto, seed_config_auto
        )
        self.architect = architect
        self.total_trials = total_trials
        self.comment = comment

        # import TrainedModel definition
        module_path, class_name = split_module_name(trained_model_table)
        self.trained_model_table = dynamic_import(module_path, class_name)

    @staticmethod
    def get_fixed_params(dataset_config, model_config, trainer_config):
        """
        Returs a single dictionary including the fixed parameters for dataset, model, and trainer.

        Args:
            dataset_config (dict): dictionary of arguments for dataset function that are fixed
            model_config (dict): dictionary of arguments for model function that are fixed
            trainer_config (dict): dictionary of arguments for trainer function that are fixed

        Returns:
            dict: A dictionary of dictionaries where keys are dataset, model, and trainer and the values are the corresponding
            dictionary of fixed arguments.
        """
        return dict(dataset=dataset_config, model=model_config, trainer=trainer_config)

    @staticmethod
    def get_auto_params(dataset_config_auto, model_config_auto, trainer_config_auto, seed_config_auto):
        """
        Returns the parameters, which are to be randomly sampled, in a list.
        Here we followed the same convention as in the Bayesian class, to have the API as similar as possible.

        Args:
            dataset_config_auto (dict): dictionary of arguments for dataset function that are to be randomly sampled
            model_config_auto (dict): dictionary of arguments for model function that are to be randomly sampled
            trainer_config_auto (dict): dictionary of arguments for trainer function that are to be randomly sampled

        Returns:
            list: list of dictionaries where each dictionary specifies a single parameter to be randomly sampled.
        """
        dataset_params = []
        for k, v in dataset_config_auto.items():
            dd = {"name": "dataset.{}".format(k)}
            dd.update(v)
            dataset_params.append(dd)

        model_params = []
        for k, v in model_config_auto.items():
            dd = {"name": "model.{}".format(k)}
            dd.update(v)
            model_params.append(dd)

        trainer_params = []
        for k, v in trainer_config_auto.items():
            dd = {"name": "trainer.{}".format(k)}
            dd.update(v)
            trainer_params.append(dd)

        seed_params = []
        for k, v in seed_config_auto.items():
            dd = {"name": "seed.{}".format(k)}
            dd.update(v)
            seed_params.append(dd)

        return dataset_params + model_params + trainer_params + seed_params

    @staticmethod
    def _combine_params(auto_params, fixed_params):
        """
        Combining the auto and fixed parameters (to have a single object representing all the arguments used for a specific function)

        Args:
            auto_params (dict): dictionary of to-be-sampled parameters, i.e. A dictionary of dictionaries where keys are dataset,
            model, and trainer and the values are the corresponding dictionary of to-be-sampled arguments.
            fixed_params (dict): dictionary of fixed parameters, i.e. A dictionary of dictionaries where keys are dataset, model,
            and trainer and the values are the corresponding dictionary of fixed arguments.

        Returns:
            dict: dictionary of parameters (fixed and to-be-sampled), i.e. A dictionary of dictionaries where keys are dataset,
            model, and trainer and the values are the corresponding dictionary of arguments.
        """
        keys = ["dataset", "model", "trainer", "seed"]
        params = {}
        for key in keys:
            params[key] = fixed_params[key] if key in fixed_params else {}
            params[key].update(auto_params[key])

        return {key: params[key] for key in keys}

    @staticmethod
    def _split_config(params):
        """
        Reverses the operation of `get_auto_params` (from a list of parameters (ax-friendly format) to a dictionary of
        dictionaries where keys are dataset, model, and trainer and the values are a dictionary of the corresponding arguments)

        Args:
            params (dict): list of dictionaries where each dictionary specifies a single parameter to be sampled.

        Returns:
            dict: A dictionary of dictionaries where keys are dataset, model, and trainer and the values are the corresponding
            dictionary of to-be-sampled arguments.
        """
        config = dict(dataset={}, model={}, trainer={}, seed={}, others={})
        for k, v in params.items():
            config[k.split(".")[0]][k.split(".")[1]] = v

        return config

    def train_evaluate(self, auto_params):
        """
        For a given set of parameters, add an entry to the corresponding tables, and populated the trained model
        table for that specific entry.

        Args:
            auto_params (dict): list of dictionaries where each dictionary specifies a single parameter to be sampled.

        """
        config = self._combine_params(self._split_config(auto_params), self.fixed_params)

        # insert the stuff into their corresponding tables
        seed = config["seed"]["seed"]
        if not dict(seed=seed) in self.trained_model_table().seed_table():
            self.trained_model_table().seed_table().insert1(dict(seed=seed))

        dataset_hash = make_hash(config["dataset"])
        entry_exists = {
            "dataset_fn": "{}".format(self.fns["dataset"])
        } in self.trained_model_table().dataset_table() and {
            "dataset_hash": "{}".format(dataset_hash)
        } in self.trained_model_table().dataset_table()
        if not entry_exists:
            self.trained_model_table().dataset_table().add_entry(
                self.fns["dataset"],
                config["dataset"],
                dataset_fabrikant=self.architect,
                dataset_comment=self.comment,
            )

        model_hash = make_hash(config["model"])
        entry_exists = {"model_fn": "{}".format(self.fns["model"])} in self.trained_model_table().model_table() and {
            "model_hash": "{}".format(model_hash)
        } in self.trained_model_table().model_table()
        if not entry_exists:
            self.trained_model_table().model_table().add_entry(
                self.fns["model"],
                config["model"],
                model_fabrikant=self.architect,
                model_comment=self.comment,
            )

        trainer_hash = make_hash(config["trainer"])
        entry_exists = {
            "trainer_fn": "{}".format(self.fns["trainer"])
        } in self.trained_model_table().trainer_table() and {
            "trainer_hash": "{}".format(trainer_hash)
        } in self.trained_model_table().trainer_table()
        if not entry_exists:
            self.trained_model_table().trainer_table().add_entry(
                self.fns["trainer"],
                config["trainer"],
                trainer_fabrikant=self.architect,
                trainer_comment=self.comment,
            )

        # get the primary key values for all those entries
        restriction = (
            f'seed in ("{seed}")',
            'dataset_fn in ("{}")'.format(self.fns["dataset"]),
            'dataset_hash in ("{}")'.format(dataset_hash),
            'model_fn in ("{}")'.format(self.fns["model"]),
            'model_hash in ("{}")'.format(model_hash),
            'trainer_fn in ("{}")'.format(self.fns["trainer"]),
            'trainer_hash in ("{}")'.format(trainer_hash),
        )

        # populate the table for those primary keys
        self.trained_model_table().populate(*restriction)

    def gen_params_value(self):
        """
        Generates new values (samples randomly) for each parameter.

        Returns:
            dict: A dictionary containing the parameters whose values should be sampled.
        """
        np.random.seed(None)
        auto_params_val = {}
        for param in self.auto_params:
            if param["type"] == "fixed":
                auto_params_val.update({param["name"]: param["value"]})
            elif param["type"] == "choice":
                auto_params_val.update({param["name"]: np.random.choice(param["values"])})
            elif param["type"] == "range":
                if "log_scale" in param and param["log_scale"]:
                    auto_params_val.update({param["name"]: loguniform.rvs(*param["bounds"])})
                else:
                    auto_params_val.update({param["name"]: np.random.uniform(*param["bounds"])})
            elif param["type"] == "int":
                auto_params_val.update({param["name"]: np.random.randint(np.iinfo(np.int32).max)})

        return auto_params_val

    def run(self):
        """
        Runs the random hyperparameter search, for as many trials as specified.
        """
        for _ in range(self.total_trials):
            self.train_evaluate(self.gen_params_value())
