import configparser

class ExperimentRoutine():
	def __init__(self, exp_list, config_file):
		self.exp_list = exp_list
		self.config = configparser.ConfigParser()
		self.config.read(config_file)

	def perform_experiments(self):
		for exp in self.exp_list:
			print('Performing experiment: {}'.format(exp['exp_name']))
			dt = exp['Dataclass'](**dict(self.config[exp['exp_name']]))
			exp = exp['ExperimentClass'](dataclass=dt)

			exp.setup()
			exp.perform_experiment()

class BaseExperimentClass():

	def __init__(self) -> None:
		# declare the classes used in this experiment
		self.network_class = None
		self.dataset_class = None

	def setup(self, dataclass):
		# setup all class variables from dataclass and variables needed for instantiate the model
		raise NotImplementedError

	def perform_experiment(self):
		# perform the experiment
		raise NotImplementedError

	def instance_model(self):
		# instantiate the experiment's NN model
		raise NotImplementedError
