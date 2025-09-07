# Replace the evaluate_recommender_enhanced function with this optimized version:

@st.cache_data
def evaluate_recommender_enhanced_fast(train_rules, test_user_games_dict, fallback_map, k_values=[5, 10, 20], max_users=100):
    """FASTEST evaluation - with user limit slider"""
    if not test_user_games_dict:
        return {}
    
    min_games_for_eval = 5
    eligible_test_users = {uid: games for uid, games in test_user_games_dict.items() 
                          if len(games) >= min_games_for_eval}
    
    if len(eligible_test_users) == 0:
        return {}
    
    # Use the slider value for max users
    max_eval_users = min(max_users, len(eligible_test_users))
    eval_user_ids = np.random.choice(list(eligible_test_users.keys()), max_eval_users, replace=False)
    
    evaluation_results = {}
    for k in k_values:
        evaluation_results[k] = {'precision_scores': [], 'recall_scores': [], 'f1_scores': []}
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, user_id in enumerate(eval_user_ids):
        # Update progress more frequently for smaller datasets
        progress = (i + 1) / len(eval_user_ids)
        progress_bar.progress(progress)
        status_text.text(f"Evaluating... {i+1}/{len(eval_user_ids)} users ({progress:.1%})")
        
        user_games_list = list(eligible_test_users[user_id])
        user_train_games, user_test_games = train_test_split(user_games_list, test_size=0.3, random_state=42)
        
        if len(user_train_games) == 0 or len(user_test_games) == 0:
            continue
            
        for k in k_values:
            try:
                # Use the simple list function for speed
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

# Update the Performance Evaluation tab section:

    with tab4:
        st.header("Performance Evaluation")
        st.write("This evaluation splits test users' games into train/test sets and measures how well the recommender predicts the held-out games.")
        
        # Evaluation configuration - Enhanced with user slider
        col1, col2, col3 = st.columns(3)
        
        with col1:
            eval_k_values = st.multiselect(
                "Select K values for evaluation:",
                [5, 10, 15, 20, 25, 30],
                default=[5, 10, 20]
            )
        
        with col2:
            # Add the user amount slider here
            max_eval_users = st.slider(
                "Max users to evaluate:",
                min_value=10,
                max_value=500,
                value=100,
                step=10,
                help="More users = more accurate but slower evaluation"
            )
        
        with col3:
            # Time estimate based on user count
            estimated_time = max_eval_users * len(eval_k_values) * 0.05  # rough estimate in seconds
            st.metric("Estimated Time", f"{estimated_time:.0f}s")
        
        # Quick evaluation presets
        st.subheader("Quick Evaluation Presets")
        preset_col1, preset_col2, preset_col3 = st.columns(3)
        
        with preset_col1:
            if st.button("🚀 Quick Test (25 users)", key="quick_eval"):
                max_eval_users = 25
                eval_k_values = [5, 10]
                run_evaluation = True
        
        with preset_col2:
            if st.button("⚡ Fast Eval (50 users)", key="fast_eval"):
                max_eval_users = 50
                eval_k_values = [5, 10, 20]
                run_evaluation = True
                
        with preset_col3:
            if st.button("📊 Full Eval (200 users)", key="full_eval"):
                max_eval_users = 200
                eval_k_values = [5, 10, 20]
                run_evaluation = True
        
        # Main evaluation button
        run_evaluation = st.button("🎯 Run Custom Evaluation", type="primary") or locals().get('run_evaluation', False)
        
        if run_evaluation and eval_k_values:
            if not test_user_games:
                st.error("No test users available for evaluation.")
            else:
                available_users = len([uid for uid, games in test_user_games.items() if len(games) >= 5])
                actual_users = min(max_eval_users, available_users)
                
                st.info(f"Starting evaluation with {actual_users} users (out of {available_users} available)...")
                
                start_time = time.time()
                
                with st.spinner(f"Running evaluation on {actual_users} users... This should take ~{actual_users * len(eval_k_values) * 0.05:.0f} seconds."):
                    evaluation_results = evaluate_recommender_enhanced_fast(
                        rules, test_user_games, genre_fallback, eval_k_values, max_eval_users
                    )
                
                end_time = time.time()
                
                if evaluation_results:
                    st.success(f"Evaluation completed in {end_time - start_time:.1f} seconds!")
                    
                    # Display results (rest remains the same as your original code)
                    # ... (keep all the existing visualization and analysis code)
                    
                else:
                    st.error("Evaluation failed to produce results. Check your data and try again.")
        
        elif run_evaluation and not eval_k_values:
            st.warning("Please select at least one K value for evaluation.")

# Additional optimization: Add this to imports at the top
import time

# And add this helper function for even faster evaluation on small datasets:

def evaluate_recommender_lightning(train_rules, test_user_games_dict, fallback_map, k=5, max_users=25):
    """Lightning fast evaluation for quick testing - single K value only"""
    if not test_user_games_dict:
        return None
    
    eligible_test_users = {uid: games for uid, games in test_user_games_dict.items() 
                          if len(games) >= 5}
    
    if len(eligible_test_users) == 0:
        return None
    
    eval_user_ids = np.random.choice(list(eligible_test_users.keys()), 
                                   min(max_users, len(eligible_test_users)), replace=False)
    
    scores = []
    
    for user_id in eval_user_ids:
        user_games_list = list(eligible_test_users[user_id])
        user_train_games, user_test_games = train_test_split(user_games_list, test_size=0.3, random_state=42)
        
        if len(user_train_games) == 0 or len(user_test_games) == 0:
            continue
            
        try:
            recommendations = recommend_games_list_only(user_train_games, train_rules, fallback_map, top_n=k)
            
            if len(recommendations) > 0:
                rec_set = set(normalize_name(game) for game in recommendations)
                test_set = set(normalize_name(game) for game in user_test_games)
                
                true_positives = len(rec_set.intersection(test_set))
                precision = true_positives / len(rec_set)
                recall = true_positives / len(test_set) if len(test_set) > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                scores.append({'precision': precision, 'recall': recall, 'f1': f1})
                
        except Exception:
            continue
    
    if scores:
        return {
            'precision': np.mean([s['precision'] for s in scores]),
            'recall': np.mean([s['recall'] for s in scores]),
            'f1': np.mean([s['f1'] for s in scores]),
            'evaluated_users': len(scores)
        }
    return None
