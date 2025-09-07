with tab2:
        st.header("System Analytics")
        
        # Get rules from session state
        rules_data = st.session_state.get('rules', pd.DataFrame())
        
        if not rules_data.empty:
            # Add evaluation toggle
            run_evaluation = st.checkbox("Run Systimport streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
import os
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
from collections import defaultdict, Counter
import gc
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Steam Game Recommender",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .game-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
        color: black;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .recommendation-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# Utility functions
def normalize_name(name):
    """Remove symbols but keep spaces and alphanumeric characters, convert to lowercase."""
    normalized = re.sub(r'[^a-zA-Z0-9\s]', '', str(name))
    normalized = ' '.join(normalized.lower().split())
    return normalized

def format_game_name(game_name):
    """Convert normalized name to title case"""
    if not game_name or game_name.strip() == "":
        return ""
    return game_name.title()

# Data loading and processing functions
@st.cache_data
def load_and_process_data(sample_size=25000, min_games=3):
    """Load and process Steam data"""
    if not os.path.exists("steam-200k.csv"):
        st.error("steam-200k.csv file not found. Please upload the dataset.")
        return None, None, None
    
    try:
        users = pd.read_csv("steam-200k.csv", header=None,
                           names=["UserID", "Game", "Behavior", "Hours", "Other"])
        
        # Filter and sample
        users = users[users["Behavior"].isin(["play", "purchase"])]
        unique_users = users['UserID'].unique()
        if len(unique_users) > sample_size:
            sampled_users = np.random.choice(unique_users, sample_size, replace=False)
            users = users[users['UserID'].isin(sampled_users)]
        
        # Create user transactions
        user_games = defaultdict(set)
        for _, row in users.iterrows():
            user_games[row['UserID']].add(normalize_name(row['Game']))
        
        # Keep only users with enough games
        user_games = {uid: games for uid, games in user_games.items() if len(games) >= min_games}
        
        # Train/test split
        all_user_ids = list(user_games.keys())
        train_user_ids, test_user_ids = train_test_split(all_user_ids, test_size=0.3, random_state=42)
        
        # Build train transactions
        train_transactions = []
        for user_id in train_user_ids:
            if user_id in user_games:
                train_transactions.append(list(user_games[user_id]))
        
        # Limit to top games
        all_games = [game for games in train_transactions for game in games]
        top_n = 5000
        top_games = set([g for g, _ in Counter(all_games).most_common(top_n)])
        
        train_transactions = [
            [g for g in games if g in top_games]
            for games in train_transactions
        ]
        
        test_user_games = {
            uid: [g for g in user_games[uid] if g in top_games]
            for uid in test_user_ids
            if len([g for g in user_games[uid] if g in top_games]) >= min_games
        }
        
        return train_transactions, test_user_games, top_games
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

@st.cache_data
def generate_association_rules(train_transactions):
    """Generate association rules"""
    if not train_transactions:
        return pd.DataFrame()
    
    try:
        te = TransactionEncoder()
        te_ary = te.fit(train_transactions).transform(train_transactions)
        df_train = pd.DataFrame(te_ary, columns=te.columns_)
        
        rule_configs = [
            (0.01, 0.4, 0.95, 1.1, 6.0),
            (0.02, 0.5, 0.9, 1.2, 5.0),
            (0.03, 0.6, 0.85, 1.5, 4.0),
            (0.05, 0.7, 0.8, 2.0, 3.5),
        ]
        
        all_rules = []
        
        for min_sup, min_conf, max_conf, min_lift, max_lift in rule_configs:
            try:
                frequent_itemsets = fpgrowth(df_train, min_support=min_sup, use_colnames=True, max_len=4)
                
                if len(frequent_itemsets) > 0:
                    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf)
                    
                    rules = rules[
                        (rules['confidence'] <= max_conf) &
                        (rules['lift'] >= min_lift) &
                        (rules['lift'] <= max_lift) &
                        (rules['support'] >= min_sup)
                    ]
                    
                    if len(rules) > 0:
                        all_rules.append(rules)
                        
            except Exception:
                continue
        
        if all_rules:
            combined_rules = pd.concat(all_rules, ignore_index=True).drop_duplicates(subset=['antecedents', 'consequents'])
            return combined_rules
        else:
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error generating rules: {e}")
        return pd.DataFrame()

@st.cache_data
def load_game_metadata():
    """Load game metadata for genre fallback"""
    if not os.path.exists("steam.csv"):
        st.warning("steam.csv not found. Genre-based recommendations will be limited.")
        return {}, {}
    
    try:
        steam_meta = pd.read_csv("steam.csv")
        game_data = {}
        genre_to_games = defaultdict(set)
        
        for _, row in steam_meta.iterrows():
            try:
                if pd.isna(row['name']) or str(row['name']).strip() == '':
                    continue
                    
                game_name = normalize_name(row['name'])
                if not game_name or game_name.strip() == '':
                    continue
                
                genres = set()
                if pd.notna(row['genres']) and str(row['genres']).strip() != '':
                    genre_list = [g.strip() for g in str(row['genres']).split(';') if g.strip()]
                    genres = set(genre_list[:5])
                
                game_data[game_name] = {
                    "name": str(row['name']),
                    "genres": genres
                }
                
                for genre in genres:
                    if genre and len(genre_to_games[genre]) < 50:
                        genre_to_games[genre].add(game_name)
                        
            except Exception:
                continue
        
        # Create fallback mapping
        genre_fallback = {}
        for game_name, info in game_data.items():
            similar_games = set()
            for genre in info['genres']:
                if genre in genre_to_games:
                    similar_in_genre = genre_to_games[genre] - {game_name}
                    similar_games.update(list(similar_in_genre)[:15])
                    if len(similar_games) > 60:
                        similar_games = set(list(similar_games)[:60])
                        break
            
            if similar_games:
                genre_fallback[game_name] = similar_games
        
        return game_data, genre_fallback
        
    except Exception as e:
        st.error(f"Error loading metadata: {e}")
        return {}, {}

def recommend_games_enhanced(user_games, rules, fallback_map=None, game_data=None, top_n=5):
    """Enhanced recommendation function"""
    user_game_set = set(normalize_name(g) for g in user_games)
    
    # Association Rules
    association_recommendations = defaultdict(float)
    rule_matches = defaultdict(list)
    
    if not rules.empty:
        for idx, rule in rules.iterrows():
            antecedents = set(rule['antecedents'])
            consequents = set(rule['consequents'])
            
            if antecedents.issubset(user_game_set):
                rule_strength = (
                    rule['confidence'] * 0.5 +
                    min(rule['lift'], 4.0) * 0.3 +
                    (rule['support'] * 10) * 0.2
                )
                
                for consequent in consequents:
                    if consequent not in user_game_set:
                        association_recommendations[consequent] += rule_strength
                        rule_matches[consequent].append({
                            'antecedents': list(antecedents),
                            'confidence': rule['confidence'],
                            'lift': rule['lift'],
                            'support': rule['support'],
                            'strength': rule_strength
                        })
    
    # Genre Fallback
    genre_recommendations = defaultdict(float)
    if len(association_recommendations) < top_n * 2 and fallback_map and game_data:
        for user_game in user_game_set:
            if user_game in fallback_map:
                similar_games = fallback_map[user_game] - user_game_set - set(association_recommendations.keys())
                
                for similar_game in similar_games:
                    if user_game in game_data and similar_game in game_data:
                        shared_genres = len(game_data[user_game]["genres"] & game_data[similar_game]["genres"])
                        if shared_genres > 0:
                            genre_score = shared_genres * 0.1
                            genre_recommendations[similar_game] += genre_score
    
    # Combine recommendations
    all_recommendations = {}
    
    for game, score in association_recommendations.items():
        all_recommendations[game] = {
            'score': score,
            'method': 'association_rules',
            'rule_count': len(rule_matches[game]),
            'supporting_rules': rule_matches[game][:3]
        }
    
    remaining_slots = max(0, top_n - len(all_recommendations))
    if remaining_slots > 0:
        sorted_genre_recs = sorted(genre_recommendations.items(), key=lambda x: x[1], reverse=True)
        
        for game, genre_score in sorted_genre_recs[:remaining_slots * 2]:
            if game not in all_recommendations:
                all_recommendations[game] = {
                    'score': genre_score,
                    'method': 'genre_fallback',
                    'rule_count': 0,
                    'supporting_rules': []
                }
    
    sorted_recommendations = sorted(all_recommendations.items(), key=lambda x: x[1]['score'], reverse=True)
    
    return sorted_recommendations[:top_n]

# Main Streamlit App
def main():
    st.markdown('<h1 class="main-header">🎮 Steam Game Recommender System</h1>', unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    sample_size = st.sidebar.slider("Sample Size", 1000, 50000, 25000, 1000)
    min_games = st.sidebar.slider("Minimum Games per User", 2, 10, 3)
    top_n = st.sidebar.slider("Number of Recommendations", 3, 20, 5)
    
    # Cache management
    st.sidebar.subheader("🗂️ Cache Management")
    
    # Show cache status
    rules_cached = os.path.exists("enhanced_association_rules.pkl")
    eval_cached = os.path.exists("evaluation_results.pkl")
    
    st.sidebar.info(f"Rules Cache: {'✅ Available' if rules_cached else '❌ Missing'}")
    st.sidebar.info(f"Evaluation Cache: {'✅ Available' if eval_cached else '❌ Missing'}")
    
    # Cache control buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🔄 Reload Data"):
            st.cache_data.clear()
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear Cache"):
            try:
                if os.path.exists("enhanced_association_rules.pkl"):
                    os.remove("enhanced_association_rules.pkl")
                if os.path.exists("evaluation_results.pkl"):
                    os.remove("evaluation_results.pkl")
                st.cache_data.clear()
                st.sidebar.success("Cache cleared!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error clearing cache: {e}")
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Load data
    with st.spinner("Loading and processing data..."):
        train_transactions, test_user_games, top_games = load_and_process_data(sample_size, min_games)
        
        if train_transactions is None:
            st.stop()
        
        rules = generate_association_rules(train_transactions)
        game_data, genre_fallback = load_game_metadata()
        st.session_state.data_loaded = True
    
    # Display data statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Training Transactions", len(train_transactions))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Association Rules", len(rules))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Games with Metadata", len(game_data))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Genre Fallback Games", len(genre_fallback))
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Get Recommendations", "📊 System Analytics", "🔍 Explore Rules"])
    
    with tab1:
        st.header("Get Game Recommendations")
        
        # Input methods
        input_method = st.radio("Choose input method:", 
                               ["Enter games manually", "Select from popular games"])
        
        user_games = []
        
        if input_method == "Enter games manually":
            games_input = st.text_area(
                "Enter your favorite games (one per line):",
                placeholder="Counter-Strike\nDota 2\nPortal",
                height=120
            )
            if games_input:
                user_games = [game.strip() for game in games_input.split('\n') if game.strip()]
        
        else:
            # Get popular games from the dataset
            if top_games:
                popular_games = sorted(list(top_games))[:100]  # Top 100 for selection
                formatted_games = [format_game_name(game) for game in popular_games]
                
                selected_games = st.multiselect(
                    "Select your favorite games:",
                    formatted_games,
                    default=[]
                )
                user_games = [normalize_name(game) for game in selected_games]
        
        # Get recommendations
        if st.button("🚀 Get Recommendations", type="primary"):
            if not user_games:
                st.warning("Please enter at least one game!")
            else:
                with st.spinner("Generating recommendations..."):
                    recommendations = recommend_games_enhanced(
                        user_games, rules, genre_fallback, game_data, top_n
                    )
                
                if recommendations:
                    st.success(f"Found {len(recommendations)} recommendations!")
                    
                    # Display user games
                    st.subheader("Your Games:")
                    cols = st.columns(min(len(user_games), 3))
                    for i, game in enumerate(user_games):
                        with cols[i % 3]:
                            formatted_name = format_game_name(game)
                            if game in game_data:
                                formatted_name = game_data[game].get("name", formatted_name)
                            st.markdown(f'<div class="game-card">🎮 {formatted_name}</div>', 
                                      unsafe_allow_html=True)
                    
                    st.subheader("Recommended Games:")
                    
                    for i, (game, info) in enumerate(recommendations, 1):
                        formatted_name = format_game_name(game)
                        if game in game_data:
                            formatted_name = game_data[game].get("name", formatted_name)
                            genres = ", ".join(list(game_data[game]["genres"])[:3])
                        else:
                            genres = "Unknown"
                        
                        method_emoji = "🔗" if info['method'] == 'association_rules' else "🏷️"
                        method_name = "Association Rules" if info['method'] == 'association_rules' else "Genre-based"
                        
                        st.markdown(f"""
                        <div class="game-card">
                            <div class="recommendation-title">{i}. {method_emoji} {formatted_name}</div>
                            <div style="margin-top: 0.5rem; color: #666;">
                                Method: {method_name} | Score: {info['score']:.3f}<br>
                                Genres: {genres}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.error("No recommendations found. Try different games or check if the dataset contains similar games.")
    
    with tab2:
        st.header("System Analytics")
        
        if len(rules) > 0:
            # Add evaluation toggle
            run_evaluation = st.checkbox("Run System Evaluation (Precision, Recall, F1)", value=False, help="This may take some time but will show performance metrics")
            
            if run_evaluation:
                evaluation_results = evaluate_recommender_system(rules, test_user_games, genre_fallback, game_data)
                
                if evaluation_results:
                    st.subheader("📊 Performance Metrics")
                    
                    # Create columns for each k value
                    eval_cols = st.columns(3)
                    
                    for i, k in enumerate([5, 10, 20]):
                        if k in evaluation_results:
                            with eval_cols[i]:
                                metrics = evaluation_results[k]
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h3>Top {k} Recommendations</h3>
                                    <p><strong>Precision:</strong> {metrics['precision']:.3f}</p>
                                    <p><strong>Recall:</strong> {metrics['recall']:.3f}</p>
                                    <p><strong>F1-Score:</strong> {metrics['f1']:.3f}</p>
                                    <p><strong>Users Evaluated:</strong> {metrics['evaluated_users']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Explanation of metrics
                    with st.expander("📖 Understanding the Metrics"):
                        st.markdown("""
                        **Precision**: Out of all games we recommended, how many were actually relevant to the user?
                        - Higher is better (fewer false positives)
                        - Range: 0.0 to 1.0
                        
                        **Recall**: Out of all games the user actually liked, how many did we recommend?
                        - Higher is better (fewer missed opportunities)
                        - Range: 0.0 to 1.0
                        
                        **F1-Score**: Harmonic mean of precision and recall
                        - Balances both precision and recall
                        - Higher is better overall performance
                        - Range: 0.0 to 1.0
                        """)
                else:
                    st.warning("Could not generate evaluation results. Check if test data is available.")
            
            # Create visualizations
            st.subheader("📈 Association Rules Visualization")
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 6))
                scatter = ax.scatter(rules['support'], rules['confidence'], 
                                   alpha=0.6, c=rules['lift'], cmap='viridis')
                ax.set_xlabel('Support')
                ax.set_ylabel('Confidence')
                ax.set_title('Association Rules: Support vs Confidence')
                plt.colorbar(scatter, label='Lift')
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.hist(rules['confidence'], bins=30, alpha=0.7, edgecolor='black')
                ax.set_xlabel('Confidence')
                ax.set_ylabel('Frequency')
                ax.set_title('Confidence Distribution')
                st.pyplot(fig)
            
            # Rules statistics
            st.subheader("📊 Rules Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Average Confidence", f"{rules['confidence'].mean():.3f}")
            with col2:
                st.metric("Average Lift", f"{rules['lift'].mean():.3f}")
            with col3:
                st.metric("Average Support", f"{rules['support'].mean():.4f}")
        
        else:
            st.warning("No association rules generated. Try adjusting the parameters.")
    
    with tab3:
        st.header("Explore Association Rules")
        
        if len(rules) > 0:
            # Filter rules
            min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.5, 0.05)
            min_lift = st.slider("Minimum Lift", 1.0, float(rules['lift'].max()), 1.5, 0.1)
            
            filtered_rules = rules[
                (rules['confidence'] >= min_confidence) & 
                (rules['lift'] >= min_lift)
            ].head(20)
            
            if len(filtered_rules) > 0:
                st.subheader(f"Top {len(filtered_rules)} Rules")
                
                for i, (_, rule) in enumerate(filtered_rules.iterrows(), 1):
                    ant = list(rule['antecedents'])
                    con = list(rule['consequents'])
                    ant_names = [format_game_name(g) for g in ant]
                    con_names = [format_game_name(g) for g in con]
                    
                    # Enhance names with metadata if available
                    if game_data:
                        ant_names = [game_data.get(g, {"name": format_game_name(g)})["name"] for g in ant]
                        con_names = [game_data.get(g, {"name": format_game_name(g)})["name"] for g in con]
                    
                    st.markdown(f"""
                    **Rule {i}:** {', '.join(ant_names)} → {', '.join(con_names)}
                    
                    - Confidence: {rule['confidence']:.3f}
                    - Lift: {rule['lift']:.2f}
                    - Support: {rule['support']:.4f}
                    """)
                    st.markdown("---")
            else:
                st.warning("No rules match the current filters.")
        else:
            st.warning("No association rules available to explore.")
    
    # Footer
    st.markdown("---")
    st.markdown("**Note:** This system uses association rule mining and genre-based fallback for game recommendations. Results depend on the dataset and may vary based on the sample size and parameters used.")

if __name__ == "__main__":
    main()
