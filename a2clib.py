import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

class TestEverythingEnv(gym.Env):
	def __init__(self):
		self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(2), gym.spaces.Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)))
		self.observation_space = gym.spaces.Tuple((gym.spaces.Discrete(2), gym.spaces.Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)))

	def step(self, action):
		state = [1]
		return state, 0, True, None

	def reset(self):
		return [0]

def get_1d_actionspace(space):
	"""
	turn a action space into a 1d space.
	the idea behind this is that you can always re-code your spaces to one-dimensional id codes. (see https://towardsdatascience.com/beginners-guide-to-custom-environments-in-openai-s-gym-989371673952).
	code from https://github.com/openai/gym/issues/1830#issuecomment-593071394
	"""
	if isinstance(space, gym.spaces.Box):
		return int(np.prod(space.shape))
	elif isinstance(space, gym.spaces.Discrete):
		return int(space.n)
	elif isinstance(space, gym.spaces.Tuple):
		return int(np.prod([get_1d_actionspace(s) for s in space.spaces]))
	elif isinstance(space, gym.spaces.Dict):
		return int(np.prod([get_1d_actionspace(s) for s in space.spaces.values()]))
	elif isinstance(space, gym.spaces.MultiBinary):
		return int(space.n)
	elif isinstance(space, gym.spaces.MultiDiscrete):
		return int(np.prod(space.shape))
	else:
		raise NotImplementedError


class Critic(tf.keras.Model):
	def __init__(self, d_neurons = [2048, 1536], d_activate = "relu"):
		super().__init__(self)
		self.real_model = []
		for neurons in d_neurons:
			self.real_model.append(tf.keras.layers.Dense(neurons, activation = d_activate))
		self.real_model.append(tf.keras.layers.Dense(1, activation = None))

	def call(self, input_data):
		rv = input_data
		for layer in self.real_model:
			rv = layer(rv)
		return rv

class Actor(tf.keras.Model):
	def __init__(self, env_actions, d_neurons = [2048, 1536], d_activate = "relu"):
		super().__init__(self)
		self.real_model = []
		for neurons in d_neurons:
			self.real_model.append(tf.keras.layers.Dense(neurons, activation = d_activate))
		self.real_model.append(tf.keras.layers.Dense(env_actions, activation = "softmax"))

	def call(self, input_data):
		rv = input_data
		for layer in self.real_model:
			rv = layer(rv)
		return rv

class Agent(object):
	def __init__(self, env, c_neurons = [2048, 1536], c_activate = "relu", a_neurons = [2048, 1536], a_activate = "relu", gamma = 0.99, learning_rate = 5e-6):
		super().__init__()
		self.gamma = gamma
		self.a_opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)
		self.c_opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)
		self.actor = Actor(get_1d_actionspace(env.action_space), d_neurons = a_neurons, d_activate = a_activate)
		self.critic = Critic(d_neurons = a_neurons, d_activate = a_activate)

	def act(self, state):
		prob = self.actor(np.array([state]))
		prob = prob.numpy()
		dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
		action = dist.sample()
		return int(action.numpy()[0])

	def actor_loss(self, prob, action, td):
		dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
		log_prob = dist.log_prob(action)
		loss = -log_prob*td
		return loss

	def learn(self, state, action, reward, next_state, done):
		state = np.array([state])
		next_state = np.array([next_state])
		with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
			p = self.actor(state, training=True)
			v =  self.critic(state,training=True)
			vn = self.critic(next_state, training=True)
			td = reward + self.gamma*vn*(1-int(done)) - v
			a_loss = self.actor_loss(p, action, td)
			c_loss = td**2
		grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
		grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)
		self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
		self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))
		return a_loss, c_loss

def check():
	env = TestEverythingEnv()
	state = env.reset()
	testagent = Agent(env)
	action = testagent.act(state)
	next_state, reward, done, _ = env.step(action)
	_, _ = testagent.learn(state, action, reward, next_state, done)
