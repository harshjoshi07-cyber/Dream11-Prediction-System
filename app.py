import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data_processor import DataProcessor
from team_optimizer import TeamOptimizer

# Set page config
st.set_page_config(
    page_title="Dream11 Team Predictor",
    page_icon="🏏",
    layout="wide"
)

# Initialize session state
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor('match.csv')
    st.session_state.data_processor.load_data()
    st.session_state.data_processor.engineer_features()
    st.session_state.data_processor.train_model()

def get_unique_players():
    """Get list of unique players from the dataset"""
    return sorted(st.session_state.data_processor.df['player'].unique())

def plot_player_form(player_name):
    """Plot player's form over last 10 matches"""
    player_data = st.session_state.data_processor.df[
        st.session_state.data_processor.df['player'] == player_name
    ].sort_values('match_id').tail(10)
    
    fig = px.line(
        player_data,
        x='match_id',
        y='dream11_score',
        title=f"{player_name}'s Last 10 Matches Performance"
    )
    fig.update_layout(
        xaxis_title="Match",
        yaxis_title="Dream11 Score",
        showlegend=False
    )
    return fig

def plot_team_composition(selected_players):
    """Plot team composition pie chart"""
    role_counts = pd.DataFrame(selected_players)['role'].value_counts()
    
    fig = px.pie(
        values=role_counts.values,
        names=role_counts.index,
        title="Team Composition by Role"
    )
    return fig

def display_team(team_data, team_number):
    """Display a single team with its details"""
    st.subheader(f"Team {team_number} (Predicted Score: {team_data['total_score']:.2f})")
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(plot_team_composition(team_data['players']))
        
    with col2:
        st.subheader("Player Details")
        for player in team_data['players']:
            role_text = player['role']
            if player.get('is_captain'):
                role_text += " (C)"
            elif player.get('is_vice_captain'):
                role_text += " (VC)"
                
            with st.expander(f"{player['name']} ({role_text})"):
                st.write(f"Predicted Score: {player['predicted_score']:.2f}")
                if player.get('is_captain'):
                    st.write("Captain (2x points)")
                elif player.get('is_vice_captain'):
                    st.write("Vice Captain (1.5x points)")
                st.plotly_chart(plot_player_form(player['name']))

def main():
    st.title("🏏 Dream11 Team Predictor")
    
    # Sidebar
    st.sidebar.header("Player Selection")
    
    # Get list of unique players
    all_players = get_unique_players()
    
    # Multi-select for players
    selected_players = st.sidebar.multiselect(
        "Select Players for Team",
        all_players,
        help="Select players you want to consider for the team"
    )
    
    # Generate team button with unique key
    generate_team = st.sidebar.button("Generate Teams", key="generate_team_button")
    
    if generate_team:
        if len(selected_players) >= 11:
            # Get data for selected players
            match_players = st.session_state.data_processor.df[
                st.session_state.data_processor.df['player'].isin(selected_players)
            ].drop_duplicates('player_id')
            
            # Predict scores for all players
            predicted_scores = {}
            for _, player in match_players.iterrows():
                player_features = match_players[match_players['player_id'] == player['player_id']].iloc[0]
                predicted_scores[player['player_id']] = st.session_state.data_processor.predict_player_score(
                    player_features[st.session_state.data_processor.feature_columns]
                )
            
            # Optimize team selection
            optimizer = TeamOptimizer(match_players, predicted_scores)
            teams = optimizer.solve(num_teams=3)
            
            # Display results
            st.header("Top 3 Dream11 Teams")
            
            # Display each team
            for i, team in enumerate(teams, 1):
                display_team(team, i)
                st.markdown("---")
            
            # Display team analysis
            st.header("Team Analysis")
            st.write("""
            These teams have been selected based on:
            - Recent form and consistency
            - Role-specific performance metrics
            - Team composition requirements
            - Player match-up statistics
            - Captain and Vice-captain multipliers (2x and 1.5x points respectively)
            """)
        else:
            st.error("Please select at least 11 players to generate teams.")

if __name__ == "__main__":
    main() 