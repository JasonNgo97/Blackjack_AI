import copy
import random
import pdb
import math
from game import Game, states

HIT = 0
STAND = 1
DISCOUNT = 0.95 #This is the gamma value for all value calculations

class Agent:
    def __init__(self):

        # For MC values
        self.MC_values = {} # Dictionary: Store the MC value of each state
        self.S_MC = {}      # Dictionary: Store the sum of returns in each state
        self.N_MC = {}      # Dictionary: Store the number of samples of each state
        # MC_values should be equal to S_MC divided by N_MC on each state (important for passing tests)

        # For TD values
        self.TD_values = {}  # Dictionary storing the TD value of each state
        self.N_TD = {}       # Dictionary: Store the number of samples of each state

        # For Q-learning values
        self.Q_values = {}   # Dictionary storing the Q-Learning value of each state and action
        self.N_Q = {}        # Dictionary: Store the number of samples of each state

        # Initialization of the values
        for s in states:
            self.MC_values[s] = 0
            self.S_MC[s] = 0
            self.N_MC[s] = 0
            self.TD_values[s] = 0
            self.N_TD[s] = 0
            self.Q_values[s] = [0,0] # First element is the Q value of "Hit", second element is the Q value of "Stand"
            self.N_Q[s] = 0
        # NOTE: see the comment of `init_cards()` method in `game.py` for description of game state       
        self.simulator = Game()

    # NOTE: do not modify
    # This is the policy for MC and TD learning. 
    @staticmethod
    def default_policy(state):
        user_sum = state[0]
        user_A_active = state[1]
        actual_user_sum = user_sum + user_A_active * 10
        if actual_user_sum < 14:
            return 0
        else:
            return 1

    # NOTE: do not modify
    # This is the fixed learning rate for TD and Q learning. 
    @staticmethod
    def alpha(n):
        return 10.0/(9 + n)
    def compute_MC_Value(self,episodeArray):
        sum=0
        for i in range(len(episodeArray)):
            decayFactor = DISCOUNT**i
            sum += decayFactor*episodeArray[i][1]
        return sum
    def MC_run(self, num_simulation, tester=False):
        # Perform num_simulation rounds of simulations in each cycle of the overall game loop
        print("Doing MC Run")
        #print("N_MC init: ",self.N_MC)
        #pdb.set_trace()
        for simulation in range(num_simulation):
            # Do not modify the following three lines
            if tester:
                self.tester_print(simulation, num_simulation, "MC")
            self.simulator.reset()  # Restart the simulator
            # TODO: Remove the following dummy updates and implement MC-learning
            # Note: Do not reset the simulator again in the rest of this simulation
            # Hint: Simulate a full episode using "self.simulator.simulate_sequence(...)"
            episode = self.simulator.simulate_sequence(self.default_policy)
            for i in range(len(episode)):
                state = episode[0][0]
                value = self.compute_MC_Value(episode)
                self.N_MC[state] += 1  #This is good
                self.S_MC[state] += value # Need to update it with the reward
                self.MC_values[state] = self.S_MC[state]/self.N_MC[state]
                episode.pop(0)


    
    def TD_run(self, num_simulation, tester=False):
        #Perform num_simulation rounds of simulations in each cycle of the overall game loop
        for simulation in range(num_simulation):
            # Do not modify the following three lines
            if tester:
                self.tester_print(simulation, num_simulation, "TD")
            self.simulator.reset()

            # TODO: Remove the following dummy updates and implement TD-learning
            # Note: Do not reset the simulator again in the rest of this simulation
            # Hint: You need a loop that takes one step simulation each time, until state is "None" which indicates termination

            # init_state = self.simulator.state
            # while self.simulator.game_over() == False:
            #     action = self.default_policy(init_state)
            #     next_state, reward =  self.simulator.simulate_one_step(action)
            #     future_state, future_reward =  self.simulator.simulate_one_step(action)
            #
            #     if next_state != None:
            #         new_factor = DISCOUNT*self.TD_values[future_state]
            #     else:
            #         new_factor = 0
            #     self.N_TD[future_state] += 1
            #     delta = reward + new_factor - self.TD_values[next_state]
            #     self.TD_values[next_state] = self.TD_values[next_state] + self.alpha(self.N_TD[next_state])*delta
            #     print("TD Value: ",self.TD_values[init_state])
            #
            #     init_state = next_state


            # The one belowo here works.

            # action = self.default_policy(self.simulator.state)
            # if (self.simulator.game_over() == False):
            #     result_state, reward = self.simulator.simulate_one_step(action)
            #     while(result_state != None):
            #         action = self.default_policy(self.simulator.state)
            #         new_state,new_reward = self.simulator.simulate_one_step(action)
            #         if new_state == None:
            #             new_factor = 0
            #         else:
            #             new_factor = DISCOUNT*self.TD_values[new_state]
            #         self.N_TD[result_state] += 1 # This will be used for alpha
            #         print("Result State: ",result_state,"\t New State: ",new_state)
            #         print("Reward: ",reward)
            #         print("New Factor: ",new_factor)
            #         print("Delta: ",reward+new_factor - self.TD_values[result_state])
            #         print("TD Before: ",self.TD_values[result_state])
            #         self.TD_values[result_state] = self.TD_values[result_state] + self.alpha(self.N_TD[result_state])*(reward+new_factor - self.TD_values[result_state])
            #         print("TD After: ",self.TD_values[result_state])
            #         reward = new_reward
            #         result_state = new_state


            # #There is an edge case of the state being 1
            init_state = self.simulator.state
            reward = 0
            while(init_state != None):
                action = self.default_policy(init_state)
                new_state, new_reward = self.simulator.simulate_one_step(action)
                if new_state == None:
                    new_factor = 0
                else:
                    new_factor = DISCOUNT*self.TD_values[new_state]
                delta = reward + new_factor - self.TD_values[init_state]
                self.N_TD[init_state] += 1 # This will be used for alpha
                self.TD_values[init_state] = self.TD_values[init_state] + self.alpha(self.N_TD[init_state])*delta
                init_state = new_state
                reward = new_reward
                
    def Q_run(self, num_simulation, tester=False):
        #Perform num_simulation rounds of simulations in each cycle of the overall game loop
        episolon = 0.4
        for simulation in range(num_simulation):
            # Do not modify the following three lines
            if tester:
                self.tester_print(simulation, num_simulation, "Q")
            self.simulator.reset()

            init_state = self.simulator.state
            while self.simulator.game_over() == False:
                action = self.pick_action(self.simulator.state,episolon)
                new_state, reward =  self.simulator.simulate_one_step(action)
                if new_state != None:
                    maxQ = max(self.Q_values[new_state])
                    self.N_Q[init_state] += 1
                    delta = reward + DISCOUNT*maxQ - self.Q_values[init_state][action]
                    self.Q_values[init_state][action] = self.Q_values[init_state][action] + self.alpha(self.N_Q[init_state])*delta
                    init_state = new_state
                else:
                    maxQ = 0
                    self.N_Q[init_state] += 1
                    delta = reward + DISCOUNT*maxQ - self.Q_values[init_state][action]
                    self.Q_values[init_state][action] = self.Q_values[init_state][action] + self.alpha(self.N_Q[init_state])*delta
                    init_state = new_state
                    break

    def pick_action(self, s, epsilon):
        # Replace the following random return value with the epsilon-greedy strategy
        # Hint: Generate a random number with `random.random()` and compare with epsilon
        # Hint: A random action is just `random.randint(0,1)`

        choice = random.uniform(0,1)
        # This is a random action
        if choice <= epsilon:
            return random.randint(0, 1)
        # Return the maximum Q value
        else:
            return self.Q_values[s].index(max(self.Q_values[s]))

    #Note: do not modify
    def autoplay_decision(self, state):
        hitQ, standQ = self.Q_values[state][HIT], self.Q_values[state][STAND]
        if hitQ > standQ:
            return HIT
        if standQ > hitQ:
            return STAND
        return HIT #Before Q-learning takes effect, just always HIT

    # NOTE: do not modify
    def save(self, filename):
        with open(filename, "w") as file:
            for table in [self.MC_values, self.TD_values, self.Q_values, self.S_MC, self.N_MC, self.N_TD, self.N_Q]:
                for key in table:
                    key_str = str(key).replace(" ", "")
                    entry_str = str(table[key]).replace(" ", "")
                    file.write(f"{key_str} {entry_str}\n")
                file.write("\n")

    # NOTE: do not modify
    def load(self, filename):
        with open(filename) as file:
            text = file.read()
            MC_values_text, TD_values_text, Q_values_text, S_MC_text, N_MC_text, NTD_text, NQ_text, _  = text.split("\n\n")
            
            def extract_key(key_str):
                return tuple([int(x) for x in key_str[1:-1].split(",")])
            
            for table, text in zip(
                [self.MC_values, self.TD_values, self.Q_values, self.S_MC, self.N_MC, self.N_TD, self.N_Q], 
                [MC_values_text, TD_values_text, Q_values_text, S_MC_text, N_MC_text, NTD_text, NQ_text]
            ):
                for line in text.split("\n"):
                    key_str, entry_str = line.split(" ")
                    key = extract_key(key_str)
                    table[key] = eval(entry_str)

    # NOTE: do not modify
    @staticmethod
    def tester_print(i, n, name):
        print(f"\r  {name} {i + 1}/{n}", end="")
        if i == n - 1:
            print()
