import os
import importlib
import random
import math
import json
from typing import Dict, List, Mapping, Tuple, Set
from datetime import datetime
from TensorRTS import Agent, GameRunner, GameResult, Runtime_Failure_Exception

from entity_gym.env import *
from entity_gym.runner import CliRunner

NUM_GAMES_PER_ROUND = 4
NUM_GAMES_WIN_BY_TO_VICTORY = 2
MAX_NUM_GAMES = 20
STEP_LIMIT = 100

#need even number of games 
assert(NUM_GAMES_PER_ROUND%2 == 0)

class Incompatability_Exception(Exception): 
    def __init__(self, script_root_dir : str, root_message : str, missing_script : bool, missing_name_hook : bool, missing_agent_hook : bool) -> None:
        assert(missing_script != None or missing_name_hook != None or missing_agent_hook != None)

        self.missing_name_hook = missing_name_hook
        self.missing_agent_hook = missing_agent_hook
        self.script_root_dir = script_root_dir

        self.root_message = root_message

        full_message = ''
        if missing_script:
            full_message = f'An error occurred while loading agent in directory: {script_root_dir} - Reson: Missing script file'
        elif missing_agent_hook:
            full_message = f'An error occurred while loading agent in directory: {script_root_dir} - Reason: Missing agent_hook function'
        elif missing_name_hook:
            full_message = f'An error occurred while loading agent in directory: {script_root_dir} - Reason: Missing display_name_hook function'

        super().__init__(full_message)

class Bot(): 

    def load_module(path_to_script):
        """Try and load from the provided script

        Args:
            path_to_script (_type_): Path to the script 
        """

        #load module
        spec = importlib.util.spec_from_file_location('agent_hook', path_to_script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return module
    
    def load_display_name(path_to_script) -> str:
        spec = importlib.util.spec_from_file_location('display_name_hook', path_to_script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        return module.display_name_hook()
    
    def find_bot_script(directory) -> str: 
        #find script
        bot_script = None

        for ele in os.listdir(target_dir): 
            ele_path = os.path.join(target_dir, ele)
            if 'agent.py' in os.path.basename(ele_path) and '__init__.py' not in ele_path:
                return ele_path

        return None
    
    def create_instance(self, init_observation : Observation, action_space : Dict[ActionName, ActionSpace]) -> Agent: 
        return self.agent_module.agent_hook(init_observation, action_space, os.path.abspath(os.path.join(self.path_to_agent_script, os.pardir)))

    def __init__(self, bot_dir : str) -> None:
        self.agent_module = None
        self.display_name = None
        
        path_to_agent_script = Bot.find_bot_script(bot_dir)
        if path_to_agent_script is None: 
            raise Incompatability_Exception(root_message="Failed to find bot script.", script_root_dir=bot_dir, missing_script=True, missing_agent_hook=False, missing_name_hook=True)
        
        try:
            self.agent_module = Bot.load_module(path_to_agent_script)
        except Exception as ex:
            raise Incompatability_Exception(root_message=repr(ex), script_root_dir=bot_dir, missing_script=False, missing_agent_hook=True, missing_name_hook = True)
        
        try:
            self.display_name : str = Bot.load_display_name(path_to_agent_script)
        except Exception as ex: 
            raise Incompatability_Exception(root_message=repr(ex), script_root_dir=bot_dir, missing_script=False, missing_agent_hook=False, missing_name_hook=True)

        self.path_to_agent_script = path_to_agent_script

class Matchup_Result(): 
    def __init__(self, first_bot_name : str, second_bot_name : str) -> None:
        self.first_bot_name = first_bot_name
        self.second_bot_name = second_bot_name

        self.game_records : List[GameResult] = []
        self.first_bot_win_count = 0
        self.second_bot_win_count = 0

    def add_result(self, game_result : GameResult):
        self.game_records.append(game_result)

        if game_result.player_one_win():
            self.first_bot_win_count += 1
        elif game_result.player_two_win():
            self.second_bot_win_count += 1

    def get_additional_info(self) -> str: 
        return self.game_records[0].additional_information

    def toJSON(self) -> json: 
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)
    
class Matchup(): 

    def run_game(first_bot : Bot, second_bot :  Bot, game_trace : str): 
        new_game = GameRunner(enable_printouts=False, trace_file=game_trace)
        
        #create bot instances
        action_space = new_game.game.action_space()

        player_one = None
        player_two = None
        player_one_exception = None
        player_two_exception = None

        try:
            player_one = first_bot.create_instance(new_game.get_game_observation(is_player_two=False), action_space)
        except Exception as ex: 
            player_one_exception = f'Error occurred while creating instance of bot: {first_bot.display_name}: {repr(ex)}'

        try:
            player_two = second_bot.create_instance(new_game.get_game_observation(is_player_two=True), action_space)
        except Exception as ex:
            player_two_exception = f'Error occurred while creating instance of bot: {second_bot.display_name}: {repr(ex)}'
            
        if player_one is None or player_two is None:
            if player_one is None and player_two is None:
                #both failed
                final_result = GameResult(0, 0, is_create_exception=True, additional_information=f'{player_one_exception}\n{player_two_exception}')
            elif player_one is None:
                final_result = GameResult(0, 1, is_create_exception=True, additional_information=player_one_exception)
            elif player_two is None: 
                final_result = GameResult(1, 0, is_create_exception=True, additional_information=player_two_exception)
                
            return final_result

        new_game.assign_players(player_one, player_two, first_bot.display_name, second_bot.display_name)

        try:
            new_game.run(STEP_LIMIT)
        except Runtime_Failure_Exception as ex:
            #check which bot caused the crash -- labeled as loss
            print(repr(ex.parent_exception))
            if ex.responsible_bot_is_player_one:
                results = GameResult(0, 1, f'Runtime error occurred for bot: {first_bot.display_name}: {repr(ex.parent_exception)}')
            else:
                results = GameResult(1, 0, f'Runtime error occurred for bot: {second_bot.display_name}: {repr(ex.parent_exception)}')
            return results 

        return new_game.results

    def __init__(self, first_bot : Bot, second_bot : Bot) -> None:
        self.first_bot = first_bot
        self.second_bot = second_bot
        self.final_result = Matchup_Result(first_bot.display_name, second_bot.display_name)

    def fight(self, num_rounds : int, num_rounds_must_win_by_to_win : int, bracket_dir : str) -> None: 
        assert(self.first_bot is not None and self.second_bot is not None)
        done = False

        num_rounds_played = 0

        print(f'Beginning match - {self.first_bot.display_name} ({self.first_bot.path_to_agent_script}) \n        Against - {self.second_bot.display_name} ({self.second_bot.path_to_agent_script})')
        
        game_root_dir = os.path.join(bracket_dir, f'{self.first_bot.display_name}v{self.second_bot.display_name}')
        if not os.path.isdir(game_root_dir):
            os.mkdir(game_root_dir)

        while done is False:
            for i in range(0, 2):
                if (num_rounds_played >= num_rounds and abs(self.final_result.first_bot_win_count - self.final_result.second_bot_win_count) >= num_rounds_must_win_by_to_win) or MAX_NUM_GAMES == num_rounds_played:
                    done = True
                    break

                game_trace = os.path.join(game_root_dir, f'{num_rounds_played}.txt')
                
                result = None
                if i == 0:
                    result = Matchup.run_game(self.first_bot, self.second_bot, game_trace)
                else:
                    result = Matchup.run_game(self.second_bot, self.first_bot, game_trace)
                    
                self.final_result.add_result(result)

                #one of the bots fils in creation...exit now
                if result.is_create_exception: 
                    return

                num_rounds_played += 1
        
    def stringify_print_results(self) -> List[str]: 
        results : List[str] = []
        display_string = f'{self.first_bot.display_name}(wins:{self.final_result.first_bot_win_count}) -- {self.second_bot.display_name}(wins:{self.final_result.second_bot_win_count})'
        results.append(display_string)
        if self.final_result.first_bot_win_count > self.final_result.second_bot_win_count:
            results.append('[winner]')
        else:
            loc = display_string.find('--')
            display_string = ' ' * (loc + 3) 
            display_string += '[winner]'
            results.append(display_string)

        add_info = self.final_result.get_additional_info()
        if add_info is not None: 
            results.append(f'Additional Round Info: {add_info}')
        
        return results
    
    def dump_results(self, path) -> None: 
        json_contents = self.final_result.toJSON()
        file_path = os.path.join(path, f'{self.first_bot.display_name}v{self.second_bot.display_name}', 'Matchup_Record.json')
        with open(file_path, 'w+') as file: 
            file.write(json_contents)

class Bracket(): 
    def __init__(self, game_trace_dir : str) -> None:
        self.run = False
        self.matchups : List[Matchup] = []

        self.game_trace_dir = game_trace_dir

    def add_matchup(self, first_bot : Bot, second_bot : Bot) -> None: 
        self.matchups.append(Matchup(first_bot, second_bot))

    def execute_game(self, num_rounds : int, num_rounds_must_win_by_to_win : int): 
        self.run = True

        for match in self.matchups:
            match.fight(num_rounds, num_rounds_must_win_by_to_win, self.game_trace_dir)

    def stringify_print_results(self) -> List[str]:
        assert(self.run)

        lines = []

        lines.append('Results of bracket: ')
        for match in self.matchups:
            match_result = match.stringify_print_results()
            
            #add padding
            for line in match_result: 
                lines.append(f'\t{line}')
        
        return lines
    
    def dump_results(self): 
        for match in self.matchups:  
            match.dump_results(self.game_trace_dir)

class Tournament():
    def __init__(self, bots : List[Bot], num_rounds_per_match : int, num_rounds_must_win_by : int) -> None:
        self.winner : Bot = None
        self.all_bots = bots
        self.in_bots = bots
        self.rounds : list[Bracket] = []

        self.num_rounds_per_match = num_rounds_per_match
        self.num_rounds_must_win_by = num_rounds_must_win_by

        current_datetime = datetime.now()
        current_date_time = current_datetime.strftime("%m-%d-%Y %H-%M-%S")

        # game trace directory
        game_root = os.path.join(os.getcwd(), 'games')
        if not os.path.isdir(game_root):
            os.mkdir(game_root)

        self.game_trace_dir = os.path.join(game_root, current_date_time)
        if not os.path.isdir(self.game_trace_dir): 
            os.mkdir(self.game_trace_dir)
    
    def create_next_round(in_bots : List[Bot], bracket_game_trace_dir : str) -> Bracket: 
        new_round = Bracket(bracket_game_trace_dir)

        already_selected = []
        for i in range(0, math.floor(len(in_bots)/2)): 
            selected_indicies = random.sample(range(len(in_bots)), 2)
            found_match = False

            while not found_match:
                if selected_indicies[0] not in already_selected and selected_indicies[1] not in already_selected: 
                    found_match = True

                    already_selected.append(selected_indicies[0])
                    already_selected.append(selected_indicies[1])

                    new_round.add_matchup(in_bots[selected_indicies[0]], in_bots[selected_indicies[1]])
                else:
                    selected_indicies = random.sample(range(len(in_bots)), 2)

        return new_round

    def remove_bot(self, bot_to_remove): 
        for i in range(0, len(self.in_bots)): 
            if self.in_bots[i].display_name == bot_to_remove.display_name: 
                del self.in_bots[i]
                return

    def run(self): 
        while len(self.in_bots) != 1: 
            print('Building next round')
            bracket_dir = os.path.join(self.game_trace_dir, f'Round {len(self.rounds)+1}')
            if not os.path.isdir(bracket_dir):
                os.mkdir(bracket_dir)

            self.rounds.append(Tournament.create_next_round(self.in_bots, bracket_dir))
            print('Running round')
            self.rounds[-1].execute_game(self.num_rounds_per_match, self.num_rounds_must_win_by)

            #remove losers from remaining bots
            for matchup in self.rounds[-1].matchups:
                if matchup.final_result.first_bot_win_count > matchup.final_result.second_bot_win_count:
                    self.remove_bot(matchup.second_bot)
                elif matchup.final_result.second_bot_win_count > matchup.final_result.first_bot_win_count: 
                    self.remove_bot(matchup.first_bot)
                else:
                    self.remove_bot(matchup.first_bot)
                    self.remove_bot(matchup.second_bot)

        self.winner = self.in_bots[0]

    def print_results(self) -> None: 
        results : list[str] = []
        results.append('-Beginning Tournament Print Out-')
        results.append("Tournament Settings: ")
        results.append(f'Winner: {self.winner.display_name}')
        results.append(f'\tNumber of games in each match: {self.num_rounds_per_match}')
        results.append(f'\tNumber of games win by for victory: {self.num_rounds_must_win_by}')
        results.append("Bracket Results:")

        num_rounds = len(self.rounds)
        for i in range(0, num_rounds):
            str_results = self.rounds[i].stringify_print_results()

            results.append(f'\tRound: {i+1}')
            for line in str_results: 
                results.append(f'\t\t{line}')

        for line in results:
            print(line)

        results_file = os.path.join(self.game_trace_dir, 'Results.txt')
        with open(results_file, 'w+') as result_file: 
            for line in results: 
                result_file.write(f'{line}\n')

    def dump_results(self, path): 
        for round in self.rounds: 
            round.dump_results()

if __name__ == "__main__":
    bots : List[Bot] = []
    compatibility_issues = []

    bot_root = os.path.join(os.getcwd(), "bots")

    if not os.path.isdir(bot_root): 
        print("Bot directory does not exist.")
        exit()

    for path in os.listdir(bot_root): 
        if "pycache" not in path and '.' not in path:
            bot_script = None
            target_dir = os.path.join(bot_root, path)

            try:
                new_bot = Bot(target_dir)
                bots.append(new_bot)
            except Incompatability_Exception as ex:
                compatibility_issues.append(ex)

    if len(compatibility_issues) != 0:
        print('Issues found while loading agents...')
        for issue in compatibility_issues:
            print(f'{repr(issue)}')

    if len(bots) != 0:
        tournament = Tournament(bots, NUM_GAMES_PER_ROUND, NUM_GAMES_WIN_BY_TO_VICTORY)
        tournament.run()
        tournament.print_results()
        tournament.dump_results('./')
    else:
        print('No compatible bots found for tournament')
        exit()
