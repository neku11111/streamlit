@st.cache_data
def evaluate_recommender_optimized(train_rules, test_user_games_dict, fallback_map, k_values=[5, 10, 20]):
    """Optimized evaluation function for Streamlit with minimal UI updates"""
    if not test_user_games_dict:
        return {}
    
    min_games_for_eval = 5
    eligible_test_users = {uid: games for uid, games in test_user_games_dict.items() 
                          if len(games) >= min_games_for_eval}
    
    if len(eligible_test_users) == 0:
        return {}
    
    # Reduce sample size for faster evaluation
    max_eval_users = min(300, len(eligible_test_users))  # Reduced from 600
    eval_user_ids = np.random.choice(list(eligible_test_users.keys()), max_eval_users, replace=False)
    
    evaluation_results = {}
    for k in k_values:
        evaluation_results[k] = {'precision_scores': [], 'recall_scores': [], 'f1_scores': []}
    
    # Create progress bar but update less frequently
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    users_evaluated = 0
    update_frequency = max(1, len(eval_user_ids) // 20)  # Update only 20 times total
    
    for i, user_id in enumerate(eval_user_ids):
        # Update progress only occasionally
        if i % update_frequency == 0 or i == len(eval_user_ids) - 1:
            progress = (i + 1) / len(eval_user_ids)
            progress_bar.progress(progress)
            status_text.text(f"Evaluating... {i+1}/{len(eval_user_ids)} users processed")
        
        user_games_list = list(eligible_test_users[user_id])
        user_train_games, user_test_games = train_test_split(user_games_list, test_size=0.3, random_state=42)
        
        if len(user_train_games) == 0 or len(user_test_games) == 0:
            continue
            
        for k in k_values:
            try:
                # Use the simpler recommendation function for speed
                recommendations = recommend_games_list_only(user_train_games, train_rules, fallback_map, top_n=k)
                
                if len(recommendations) == 0:
                    precision = recall = f1 = 0.0
                else:
                    rec_set = set(normalize_name(game) for game in recommendations)
                    test_set = set(normalize_name(game) for game in user_test_games)
                    
                    true_positives = len(rec_set.intersection(test_set))
                    precision = true_positives / len(rec_set) if len(rec_set) > 0 else 0.0
                    recall = true_positives / len(test_set) if len(test_set) > 0 else 0.0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                evaluation_results[k]['precision_scores'].append(precision)
                evaluation_results[k]['recall_scores'].append(recall)
                evaluation_results[k]['f1_scores'].append(f1)
                
            except Exception:
                continue
        
        users_evaluated += 1
    
    # Clean up progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Calculate averages
    final_results = {}
    for k in k_values:
        if len(evaluation_results[k]['precision_scores']) > 0:
            final_results[k] = {
                'precision': np.mean(evaluation_results[k]['precision_scores']),
                'recall': np.mean(evaluation_results[k]['recall_scores']),
                'f1': np.mean(evaluation_results[k]['f1_scores']),
                'evaluated_users': len(evaluation_results[k]['precision_scores'])
            }
    
    return final_results

# Also add this simpler function if you don't have it
def recommend_games_list_only(user_games, rules, fallback_map=None, top_n=5):
    """Simple recommendation function that returns only game names"""
    recommendations = recommend_games_enhanced(user_games, rules, fallback_map, None, top_n)
    return [game for game, _ in recommendations]
