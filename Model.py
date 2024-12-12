import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

@dataclass
class PlayerStats:
    """Player statistics configuration"""
    GOALKEEPER = {
        'Goals': -0.3,        # Negative weight for goals conceded
        'SoT%': 0.25,        # Shot stopping
        'Clr': 0.15,         # Clearances 
        'TouDefPen': 0.15,   # Penalty area control
        'AerWon%': 0.15      # Aerial ability
    }
    
    DEFENDER = {
        'Tkl': 0.2,          # Tackling
        'Int': 0.2,          # Interceptions
        'Clr': 0.15,         # Clearances
        'AerWon%': 0.15,     # Aerial duels
        'PasTotCmp%': 0.15,  # Passing accuracy
        'Blocks': 0.15       # Blocks
    }
    
    MIDFIELDER = {
        'PasTotCmp%': 0.2,   # Passing
        'PasProg': 0.2,      # Progressive passes
        'Assists': 0.15,     # Assists
        'SCA': 0.15,         # Shot creating actions  
        'PPA': 0.15,         # Passes into penalty area
        'Int': 0.15          # Interceptions
    }
    
    FORWARD = {
        'Goals': 0.3,        # Goals
        'SoT%': 0.15,       # Shot accuracy
        'G/Sh': 0.15,       # Conversion rate
        'GCA': 0.15,        # Goal creating actions
        'Assists': 0.15,     # Assists
        'ToSuc%': 0.1       # Dribbling success
    }

class TeamPerformance:
    def __init__(self, team_stats_df: pd.DataFrame):
        self.team_stats = team_stats_df
        
    def calculate_team_strength(self, team: str) -> float:
        team_data = self.team_stats[self.team_stats['Squad'] == team].iloc[0]
        
        # Normalize team metrics
        win_rate = team_data['W'] / team_data['MP']
        goal_diff_per_game = team_data['GD'] / team_data['MP']
        xg_diff_per_game = team_data['xGD'] / team_data['MP']
        
        # Calculate weighted team strength
        strength = (
            0.4 * win_rate +
            0.3 * (goal_diff_per_game / 3) + # Normalize by typical max GD
            0.3 * (xg_diff_per_game / 2)     # Normalize by typical max xGD
        )
        
        return strength

class PlayerEvaluator:
    def __init__(self, player_stats_df: pd.DataFrame, team_stats_df: pd.DataFrame):
        self.player_stats = player_stats_df
        self.team_performance = TeamPerformance(team_stats_df)
        self.stats_config = PlayerStats()
        
    def normalize_stat(self, values: pd.Series) -> pd.Series:
        """Normalize statistics to 0-1 scale"""
        min_val = values.min()
        max_val = values.max()
        if max_val == min_val:
            return pd.Series(0.5, index=values.index)
        return (values - min_val) / (max_val - min_val)
    
    def calculate_player_score(self, player: pd.Series) -> float:
        """Calculate individual player score based on position and stats"""
        position = player['Pos'][:2]
        
        if position == 'GK':
            weights = self.stats_config.GOALKEEPER
        elif position == 'DF': 
            weights = self.stats_config.DEFENDER
        elif position == 'MF':
            weights = self.stats_config.MIDFIELDER
        elif position == 'FW':
            weights = self.stats_config.FORWARD
        else:
            print(f"Warning: Unknown position {position} for player {player['Player']}")
            return 0.0
            
        score = 0
        position_group = self.player_stats[self.player_stats['Pos'].str.startswith(position)]
        
        for stat, weight in weights.items():
            if stat not in player:
                continue
                
            value = player[stat]
            if pd.notnull(value):
                normalized = self.normalize_stat(position_group[stat])[player.name]
                score += normalized * weight
                
        # Adjust score by team context
        team_strength = self.team_performance.calculate_team_strength(player['Squad'])
        final_score = score * (1 + team_strength) / 2

        
        return final_score
    
    def evaluate_all_players(self) -> pd.DataFrame:
        """Score all players in dataset"""
        self.player_stats['player_score'] = self.player_stats.apply(
            self.calculate_player_score, axis=1
        )
        return self.player_stats[['Player', 'Pos', 'Squad', 'player_score']]

class MatchPredictor:
    def __init__(self, player_evaluator: PlayerEvaluator):
        self.player_evaluator = player_evaluator
        
    def predict_match(self, team1_players: List[str], team2_players: List[str]) -> Dict:
        """Predict match outcome between two teams"""
        # Get player scores with validation
        team1_scores = []
        team2_scores = []
        
        # Calculate weighted scores by position
        for player in team1_players:
            player_data = self.player_evaluator.player_stats[
                self.player_evaluator.player_stats['Player'] == player
            ]
            if not player_data.empty:
                position = player_data['Pos'].iloc[0][:2]
                score = player_data['player_score'].iloc[0]
                # Apply position-based importance
                if position == 'GK':
                    score *= 1.2  # Goalkeeper has high impact
                elif position == 'FW':
                    score *= 1.1  # Forward has above average impact
                team1_scores.append(score)
            
        for player in team2_players:
            player_data = self.player_evaluator.player_stats[
                self.player_evaluator.player_stats['Player'] == player
            ]
            if not player_data.empty:
                position = player_data['Pos'].iloc[0][:2]
                score = player_data['player_score'].iloc[0]
                # Apply position-based importance
                if position == 'GK':
                    score *= 1.2
                elif position == 'FW':
                    score *= 1.1
                team2_scores.append(score)
                
        if not team1_scores or not team2_scores:
            raise ValueError("Not enough valid players found")
            
        # Calculate weighted team strengths
        team1_strength = np.mean(team1_scores) 
        team2_strength = np.mean(team2_scores)
        
        # Add team performance impact
        team1_squad = self.player_evaluator.player_stats[
            self.player_evaluator.player_stats['Player'].isin(team1_players)
        ]['Squad'].iloc[0]
        team2_squad = self.player_evaluator.player_stats[
            self.player_evaluator.player_stats['Player'].isin(team2_players)
        ]['Squad'].iloc[0]
        
        team1_perf = self.player_evaluator.team_performance.calculate_team_strength(team1_squad)
        team2_perf = self.player_evaluator.team_performance.calculate_team_strength(team2_squad)
        
        # Combine individual strength and team performance
        team1_final = team1_strength * (1 + team1_perf)
        team2_final = team2_strength * (1 + team2_perf)
        
        # Calculate win probability with enhanced difference
        diff = (team1_final - team2_final) * 2  # Amplify differences
        team1_win_prob = 1 / (1 + np.exp(-diff))
        
        return {
            'team1_win_prob': team1_win_prob,
            'team2_win_prob': 1 - team1_win_prob,
            'predicted_score_diff': diff,
            'team1_strength': team1_final,  # Add for debugging
            'team2_strength': team2_final   # Add for debugging
        }

class WeightTracker:
    def __init__(self):
        self.initial_weights = PlayerStats()
        self.final_weights = None
        
    def print_weight_comparison(self):
        """Print initial and final weights for each position"""
        positions = ['GOALKEEPER', 'DEFENDER', 'MIDFIELDER', 'FORWARD']
        
        for pos in positions:
            print(f"\n{pos} Position Weights:")
            print("-" * 50)
            print(f"{'Stat':<15} {'Initial':<10} {'Final':<10} {'Change':<10}")
            print("-" * 50)
            
            initial = getattr(self.initial_weights, pos)
            final = getattr(self.final_weights, pos) if self.final_weights else initial
            
            for stat in initial.keys():
                init_val = initial[stat]
                final_val = final[stat]
                change = final_val - init_val
                print(f"{stat:<15} {init_val:>8.3f}  {final_val:>8.3f}  {change:>+8.3f}")

# Modify PlayerScorer to track weights
class PlayerScorer:
    def __init__(self, stats_df: pd.DataFrame):
        self.stats_df = stats_df
        self.weight_tracker = WeightTracker()
        self.weights = self.weight_tracker.initial_weights

    def train(self):
        # Your existing training code here
        # After training, update final weights
        self.weight_tracker.final_weights = self.weights
        
    def print_weights(self):
        self.weight_tracker.print_weight_comparison()

class WeightLearner(nn.Module):
    def __init__(self, initial_weights: PlayerStats):
        super().__init__()
        
        # Convert initial weights to learnable parameters
        self.gk_weights = nn.Parameter(torch.tensor(
            [v for v in initial_weights.GOALKEEPER.values()], dtype=torch.float32))
        self.df_weights = nn.Parameter(torch.tensor(
            [v for v in initial_weights.DEFENDER.values()], dtype=torch.float32))
        self.mf_weights = nn.Parameter(torch.tensor(
            [v for v in initial_weights.MIDFIELDER.values()], dtype=torch.float32))
        self.fw_weights = nn.Parameter(torch.tensor(
            [v for v in initial_weights.FORWARD.values()], dtype=torch.float32))
        
    def forward(self, player_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        position_scores = []
        
        # Calculate average score per position with proper dimensions
        if 'GK' in player_features:
            gk_score = (self.gk_weights.unsqueeze(0) * player_features['GK']).sum(dim=1).mean()
            position_scores.append(gk_score)
            
        if 'DF' in player_features:
            df_score = (self.df_weights.unsqueeze(0) * player_features['DF']).sum(dim=1).mean()
            position_scores.append(df_score)
            
        if 'MF' in player_features:
            mf_score = (self.mf_weights.unsqueeze(0) * player_features['MF']).sum(dim=1).mean()
            position_scores.append(mf_score)
            
        if 'FW' in player_features:
            fw_score = (self.fw_weights.unsqueeze(0) * player_features['FW']).sum(dim=1).mean()
            position_scores.append(fw_score)
        
        # Stack and ensure output has proper shape [1]
        return torch.stack(position_scores).mean().unsqueeze(0)

class AdaptivePlayerScorer:
    def __init__(self, player_stats: pd.DataFrame, team_stats: pd.DataFrame):
        self.player_stats = player_stats
        self.team_stats = team_stats
        self.initial_weights = PlayerStats()
        self.model = WeightLearner(self.initial_weights)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
    def prepare_match_data(self):
        """Create training data from team stats"""
        matches = []
        
        # Create match pairs from team stats
        teams = self.team_stats['Squad'].unique()
        for i in range(len(teams)):
            for j in range(i+1, len(teams)):
                team1, team2 = teams[i], teams[j]
                
                # Get actual results from stats
                team1_stats = self.team_stats[self.team_stats['Squad'] == team1].iloc[0]
                team2_stats = self.team_stats[self.team_stats['Squad'] == team2].iloc[0]
                
                # Calculate win probability based on points
                total_points = team1_stats['Pts'] + team2_stats['Pts']
                if total_points > 0:
                    win_prob = team1_stats['Pts'] / total_points
                else:
                    win_prob = 0.5
                    
                matches.append({
                    'team1': team1,
                    'team2': team2,
                    'win_prob': win_prob
                })
                
        return matches
    
    def get_team_features(self, team: str) -> Dict[str, torch.Tensor]:
        """Extract feature tensors by position for a team"""
        team_players = self.player_stats[self.player_stats['Squad'] == team]
        features = {}
        
        # Standard formation sizes
        formation_size = {
            'GK': 1,
            'DF': 4,
            'MF': 4, 
            'FW': 2
        }
        
        for pos in ['GK', 'DF', 'MF', 'FW']:
            pos_players = team_players[team_players['Pos'].str.startswith(pos)]
            if len(pos_players) > 0:
                pos_features = []
                
                # Take top N players based on minutes played
                pos_players = pos_players.nlargest(formation_size[pos], '90s')
                
                for _, player in pos_players.iterrows():
                    if pos == 'GK':
                        stats = self.initial_weights.GOALKEEPER.keys()
                    elif pos == 'DF':
                        stats = self.initial_weights.DEFENDER.keys()
                    elif pos == 'MF':
                        stats = self.initial_weights.MIDFIELDER.keys()
                    else:
                        stats = self.initial_weights.FORWARD.keys()
                        
                    features_list = []
                    for stat in stats:
                        if stat in player and pd.notnull(player[stat]):
                            features_list.append(float(player[stat]))
                        else:
                            features_list.append(0.0)
                    pos_features.append(features_list)
                
                # Pad if needed
                while len(pos_features) < formation_size[pos]:
                    pos_features.append([0.0] * len(features_list))
                    
                features[pos] = torch.tensor(pos_features, dtype=torch.float32)
                
        return features
    
    def train(self, epochs: int = 50, patience: int = 10):
        matches = self.prepare_match_data()
        print(f"\nStarting training with {len(matches)} match pairs")
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            total_loss = self.train_epoch(matches)
            
            # Print current weights every epoch
            print(f"\nLoss: {total_loss/len(matches):.4f}")
            print("\nCurrent weights:")
            
            # Goalkeeper weights
            print("\nGoalkeeper:")
            for stat, weight in zip(self.initial_weights.GOALKEEPER.keys(), 
                                  self.model.gk_weights.detach().numpy()):
                print(f"{stat:>10}: {weight:>8.4f}")
                
            # Defender weights    
            print("\nDefender:")
            for stat, weight in zip(self.initial_weights.DEFENDER.keys(),
                                  self.model.df_weights.detach().numpy()):
                print(f"{stat:>10}: {weight:>8.4f}")
                
            # Midfielder weights
            print("\nMidfielder:")
            for stat, weight in zip(self.initial_weights.MIDFIELDER.keys(),
                                  self.model.mf_weights.detach().numpy()):
                print(f"{stat:>10}: {weight:>8.4f}")
                
            # Forward weights
            print("\nForward:")
            for stat, weight in zip(self.initial_weights.FORWARD.keys(),
                                  self.model.fw_weights.detach().numpy()):
                print(f"{stat:>10}: {weight:>8.4f}")
            
            print("-" * 50)
            
            # Early stopping check
            if total_loss < best_loss:
                best_loss = total_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    def train_epoch(self, matches):
        total_loss = 0
        matches_processed = 0
        
        for match in matches:
            if matches_processed % 10 == 0:  # Show progress every 10 matches
                print(f"\rProcessing match {matches_processed+1}/{len(matches)}", end="")
                
            self.optimizer.zero_grad()
            
            # Get team features
            team1_features = self.get_team_features(match['team1'])
            team2_features = self.get_team_features(match['team2'])
            
            # Calculate predicted probabilities with proper dimensions
            team1_score = self.model(team1_features)
            team2_score = self.model(team2_features)
            
            pred_prob = torch.sigmoid(team1_score - team2_score)
            target = torch.tensor([match['win_prob']], dtype=torch.float32)
            
            # Ensure tensors have same shape
            assert pred_prob.shape == target.shape, f"Shape mismatch: {pred_prob.shape} vs {target.shape}"
            
            # Binary cross entropy loss
            loss = F.binary_cross_entropy(pred_prob, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            matches_processed += 1
            
        print()  # New line after progress bar
        return total_loss
    
    def get_final_weights(self) -> PlayerStats:
        """Return learned weights in original format"""
        final_weights = PlayerStats()
        
        # Convert learned parameters back to dictionaries
        gk_weights = self.model.gk_weights.detach().numpy()
        df_weights = self.model.df_weights.detach().numpy()
        mf_weights = self.model.mf_weights.detach().numpy()
        fw_weights = self.model.fw_weights.detach().numpy()
        
        # Update weights
        for i, (stat, _) in enumerate(final_weights.GOALKEEPER.items()):
            final_weights.GOALKEEPER[stat] = gk_weights[i]
        for i, (stat, _) in enumerate(final_weights.DEFENDER.items()):
            final_weights.DEFENDER[stat] = df_weights[i]
        for i, (stat, _) in enumerate(final_weights.MIDFIELDER.items()):
            final_weights.MIDFIELDER[stat] = mf_weights[i]
        for i, (stat, _) in enumerate(final_weights.FORWARD.items()):
            final_weights.FORWARD[stat] = fw_weights[i]
            
        return final_weights

# Usage
if __name__ == "__main__":
    # Load data
    player_stats = pd.read_csv('player_stats.csv', sep=';', encoding='latin-1')
    team_stats = pd.read_csv('team_stats.csv', sep=';', encoding='latin-1')
    
    # Initialize evaluator and calculate scores
    evaluator = PlayerEvaluator(player_stats, team_stats)
    player_scores = evaluator.evaluate_all_players()

    predictor = MatchPredictor(evaluator)
    

    
    # Initialize predictor
    predictor = MatchPredictor(evaluator)
    
    # Example prediction
    team1 = ['Doðan Alemdar', 'Yunis Abdelhamid', 'Abner', 'Francesco Acerbi', 
             'Marcos Acuña', 'Brenden Aaronson', 'Himad Abdelli', 'Salis Abdul Samed', 
             'Matthis Abline', 'Matthis Abline', 'Zakaria Aboukhlal']

    team2 = ['Alisson', 'Tosin Adarabioyo', 'Emmanuel Agbadou', 'Felix Agu',
             'Nayef Aguerd', 'Laurent Abergel', 'Oliver Abildgaard', 'Tyler Adams',
             'Tammy Abraham', 'Mohamed Achi', 'Erling Haaland']
    
    result = predictor.predict_match(team1, team2)
    print("\nMatch prediction:")
    print(f"Team 1 strength: {result['team1_strength']:.4f}")
    print(f"Team 2 strength: {result['team2_strength']:.4f}")
    print(f"Team 1 win probability: {result['team1_win_prob']:.4f}")
    print(f"Team 2 win probability: {result['team2_win_prob']:.4f}")

    # Initialize and train adaptive scorer
    scorer = AdaptivePlayerScorer(player_stats, team_stats)
    scorer.train(epochs=100)
    
    # Get and print learned weights
    final_weights = scorer.get_final_weights()
    
    # Print comparison
    weight_tracker = WeightTracker()
    weight_tracker.initial_weights = PlayerStats()
    weight_tracker.final_weights = final_weights
    weight_tracker.print_weight_comparison()
