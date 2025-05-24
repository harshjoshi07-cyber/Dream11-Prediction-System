import pandas as pd
import numpy as np
from pulp import *

class TeamOptimizer:
    def __init__(self, player_data, predicted_scores):
        """
        Initialize team optimizer
        player_data: DataFrame with player information
        predicted_scores: Dictionary of player_id to predicted Dream11 score
        """
        self.player_data = player_data
        self.predicted_scores = predicted_scores
        self.selected_teams = []  # Store previously selected teams to avoid duplicates
        
    def create_optimization_problem(self, exclude_players=None):
        """Create and solve the optimization problem for team selection"""
        prob = LpProblem("Dream11_Team_Selection", LpMaximize)
        
        # Create binary variables for each player
        player_vars = LpVariable.dicts("player",
                                     self.player_data['player_id'],
                                     cat='Binary')
        
        # Objective: Maximize total predicted score
        prob += lpSum([self.predicted_scores[pid] * player_vars[pid] 
                      for pid in self.player_data['player_id']])
        
        # Constraints
        # 1. Exactly 11 players
        prob += lpSum([player_vars[pid] for pid in self.player_data['player_id']]) == 11
        
        # 2. Minimum 3 batsmen
        prob += lpSum([player_vars[pid] for pid in self.player_data['player_id'] 
                      if self.player_data.loc[self.player_data['player_id'] == pid, 'is_batsman'].iloc[0] == 1]) >= 3
        
        # 3. Minimum 3 bowlers
        prob += lpSum([player_vars[pid] for pid in self.player_data['player_id'] 
                      if self.player_data.loc[self.player_data['player_id'] == pid, 'is_bowler'].iloc[0] == 1]) >= 3
        
        # 4. Maximum 4 all-rounders
        prob += lpSum([player_vars[pid] for pid in self.player_data['player_id'] 
                      if self.player_data.loc[self.player_data['player_id'] == pid, 'is_allrounder'].iloc[0] == 1]) <= 4
        
        # 5. Maximum 1 wicket-keeper
        prob += lpSum([player_vars[pid] for pid in self.player_data['player_id'] 
                      if self.player_data.loc[self.player_data['player_id'] == pid, 'stump'].iloc[0] > 0]) <= 1
        
        # 6. Exclude previously selected players if specified
        if exclude_players:
            for pid in exclude_players:
                prob += player_vars[pid] == 0
        
        return prob, player_vars
        
    def solve(self, num_teams=3):
        """Solve the optimization problem and return multiple teams"""
        all_teams = []
        
        for _ in range(num_teams):
            # Get previously selected players from all teams
            exclude_players = []
            for team in all_teams:
                exclude_players.extend([p['player_id'] for p in team['players']])
            
            prob, player_vars = self.create_optimization_problem(exclude_players)
            
            # Solve the problem
            prob.solve()
            
            if prob.status != 1:  # If no solution found
                break
                
            # Get selected players
            selected_players = []
            for pid in self.player_data['player_id']:
                if value(player_vars[pid]) == 1:
                    player_info = self.player_data[self.player_data['player_id'] == pid].iloc[0]
                    selected_players.append({
                        'player_id': pid,
                        'name': player_info['player'],
                        'predicted_score': self.predicted_scores[pid],
                        'role': self._get_player_role(player_info)
                    })
            
            # Assign captain and vice-captain
            if selected_players:
                # Sort players by predicted score
                selected_players.sort(key=lambda x: x['predicted_score'], reverse=True)
                
                # Assign captain (2x points) and vice-captain (1.5x points)
                selected_players[0]['is_captain'] = True
                selected_players[0]['multiplier'] = 2.0
                selected_players[1]['is_vice_captain'] = True
                selected_players[1]['multiplier'] = 1.5
                
                # Set multiplier for other players
                for player in selected_players[2:]:
                    player['multiplier'] = 1.0
                
                # Calculate total team score with captain/vice-captain multipliers
                total_score = sum(player['predicted_score'] * player['multiplier'] 
                                for player in selected_players)
                
                all_teams.append({
                    'players': selected_players,
                    'total_score': total_score
                })
        
        return all_teams
    
    def _get_player_role(self, player_info):
        """Determine player role based on their stats"""
        if player_info['is_allrounder'] == 1:
            return 'All-rounder'
        elif player_info['is_batsman'] == 1:
            return 'Batsman'
        elif player_info['is_bowler'] == 1:
            return 'Bowler'
        else:
            return 'Wicket-keeper' 