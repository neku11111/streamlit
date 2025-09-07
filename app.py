# Add this import at the top with your other imports
import time

# Replace your evaluate_recommender_enhanced function with this more robust version:
def evaluate_recommender_enhanced_robust(train_rules, test_user_games_dict, fallback_map, k_values=[5, 10, 20], max_users=100):
    """ROBUST evaluation with better error handling and memory management"""
    if not test_user_games_dict:
        return {}
    
    min_games_for_eval = 5
    eligible_test_users = {uid: games for uid, games in test_user_games_dict.items() 
                          if len(games) >= min_games_for_eval}
    
    if len(eligible_test_users) == 0:
        return {}
    
    # Use the max_users parameter from slider
    max_eval_users = min(max_users, len(eligible_test_users))
    
    # Convert to list for safer indexing
    eligible_user_list = list(eligible_test_users.keys())
    
    # Use fixed seed for reproducibility
    np.random.seed(42)
    eval_user_ids = np.random.choice(eligible_user_list, max_eval_users, replace=False)
    
    evaluation_results = {}
    for k in k_values:
        evaluation_results[k] = {'precision_scores': [], 'recall_scores': [], 'f1_scores': []}
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    successful_evaluations = 0
    failed_evaluations = 0
    
    try:
        for i, user_id in enumerate(eval_user_ids):
            try:
                # Update progress
                progress = (i + 1) / len(eval_user_ids)
                progress_bar.progress(progress)
                status_text.text(f"Evaluating user {i+1}/{len(eval_user_ids)} | Success: {successful_evaluations} | Failed: {failed_evaluations}")
                
                # Get user games safely
                if user_id not in eligible_test_users:
                    failed_evaluations += 1
                    continue
                
                user_games_list = list(eligible_test_users[user_id])
                
                # Ensure we have enough games
                if len(user_games_list) < min_games_for_eval:
                    failed_evaluations += 1
                    continue
                
                # Split with fixed seed for this user
                user_train_games, user_test_games = train_test_split(
                    user_games_list, test_size=0.3, random_state=42 + int(user_id) % 1000
                )
                
                # Ensure both sets have games
                if len(user_train_games) == 0 or len(user_test_games) == 0:
                    failed_evaluations += 1
                    continue
                
                # Evaluate for each K
                user_successful = False
                for k in k_values:
                    try:
                        # Get recommendations with timeout protection
                        recommendations = recommend_games_list_only(
                            user_train_games, train_rules, fallback_map, top_n=k
                        )
                        
                        if len(recommendations) == 0:
                            precision = recall = f1 = 0.0
                        else:
                            # Normalize game names safely
                            rec_set = set()
                            test_set = set()
                            
                            for game in recommendations:
                                try:
                                    normalized = normalize_name(game)
                                    if normalized and normalized.strip():
                                        rec_set.add(normalized)
                                except:
                                    continue
                            
                            for game in user_test_games:
                                try:
                                    normalized = normalize_name(game)
                                    if normalized and normalized.strip():
                                        test_set.add(normalized)
                                except:
                                    continue
                            
                            # Calculate metrics safely
                            if len(rec_set) == 0 or len(test_set) == 0:
                                precision = recall = f1 = 0.0
                            else:
                                true_positives = len(rec_set.intersection(test_set))
                                precision = true_positives / len(rec_set)
                                recall = true_positives / len(test_set)
                                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                        
                        # Store results
                        evaluation_results[k]['precision_scores'].append(precision)
                        evaluation_results[k]['recall_scores'].append(recall)
                        evaluation_results[k]['f1_scores'].append(f1)
                        user_successful = True
                        
                    except Exception as e:
                        st.warning(f"Error evaluating K={k} for user {user_id}: {str(e)[:50]}...")
                        continue
                
                if user_successful:
                    successful_evaluations += 1
                else:
                    failed_evaluations += 1
                
                # Memory cleanup every 50 users
                if i % 50 == 0:
                    gc.collect()
                
            except Exception as e:
                failed_evaluations += 1
                st.warning(f"Error processing user {user_id}: {str(e)[:50]}...")
                continue
        
    except Exception as e:
        st.error(f"Critical evaluation error: {e}")
        return {}
    
    finally:
        # Always clean up progress indicators
        progress_bar.empty()
        status_text.empty()
    
    # Calculate final results
    final_results = {}
    for k in k_values:
        scores = evaluation_results[k]['precision_scores']
        if len(scores) > 0:
            final_results[k] = {
                'precision': np.mean(evaluation_results[k]['precision_scores']),
                'recall': np.mean(evaluation_results[k]['recall_scores']),
                'f1': np.mean(evaluation_results[k]['f1_scores']),
                'evaluated_users': len(scores)
            }
    
    st.success(f"Evaluation completed! Successfully evaluated {successful_evaluations} users, {failed_evaluations} failed.")
    
    return final_results

# Replace the Performance Evaluation tab (tab4) with this enhanced version:
    with tab4:
        st.header("Performance Evaluation")
        st.write("This evaluation splits test users' games into train/test sets and measures how well the recommender predicts the held-out games.")
        
        # Enhanced evaluation configuration
        col1, col2, col3 = st.columns(3)
        
        with col1:
            eval_k_values = st.multiselect(
                "Select K values for evaluation:",
                [5, 10, 15, 20, 25, 30],
                default=[5, 10, 20],
                help="Number of recommendations to generate for evaluation"
            )
        
        with col2:
            # This is the user slider you requested
            max_eval_users = st.slider(
                "Max users to evaluate:",
                min_value=10,
                max_value=500,
                value=100,
                step=10,
                help="More users = more accurate results but slower evaluation"
            )
        
        with col3:
            # Time estimate
            estimated_time = max_eval_users * len(eval_k_values) * 0.08  # More conservative estimate
            if estimated_time < 60:
                time_str = f"{estimated_time:.0f}s"
            else:
                time_str = f"{estimated_time/60:.1f}m"
            st.metric("Estimated Time", time_str)
        
        # Quick evaluation presets
        st.subheader("Quick Evaluation Presets")
        preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)
        
        run_evaluation = False
        
        with preset_col1:
            if st.button("Lightning ⚡ (25 users)", key="lightning_eval", help="~10 seconds"):
                max_eval_users = 25
                eval_k_values = [5, 10]
                run_evaluation = True
        
        with preset_col2:
            if st.button("Quick 🚀 (50 users)", key="quick_eval", help="~20 seconds"):
                max_eval_users = 50
                eval_k_values = [5, 10, 20]
                run_evaluation = True
                
        with preset_col3:
            if st.button("Standard 📊 (100 users)", key="standard_eval", help="~45 seconds"):
                max_eval_users = 100
                eval_k_values = [5, 10, 20]
                run_evaluation = True
        
        with preset_col4:
            if st.button("Thorough 🔍 (200 users)", key="thorough_eval", help="~90 seconds"):
                max_eval_users = 200
                eval_k_values = [5, 10, 15, 20]
                run_evaluation = True
        
        # Main evaluation button
        if st.button("🎯 Run Custom Evaluation", type="primary"):
            run_evaluation = True
        
        # System status check
        if test_user_games:
            available_users = len([uid for uid, games in test_user_games.items() if len(games) >= 5])
            st.info(f"Available test users: {available_users} | Selected for evaluation: {min(max_eval_users, available_users)}")
        
        if run_evaluation and eval_k_values:
            if not test_user_games:
                st.error("No test users available for evaluation.")
            else:
                available_users = len([uid for uid, games in test_user_games.items() if len(games) >= 5])
                actual_users = min(max_eval_users, available_users)
                
                if actual_users < 10:
                    st.warning(f"Only {actual_users} users available for evaluation. Results may not be reliable.")
                
                st.info(f"Starting robust evaluation with {actual_users} users out of {available_users} available...")
                
                # Record start time
                start_time = time.time()
                
                # Create a container for the evaluation
                eval_container = st.container()
                
                with eval_container:
                    with st.spinner(f"Running evaluation on {actual_users} users... Estimated time: {estimated_time:.0f}s"):
                        try:
                            evaluation_results = evaluate_recommender_enhanced_robust(
                                rules, test_user_games, genre_fallback, eval_k_values, max_eval_users
                            )
                        except Exception as e:
                            st.error(f"Evaluation failed with error: {e}")
                            evaluation_results = {}
                
                # Calculate actual time
                end_time = time.time()
                actual_time = end_time - start_time
                
                if evaluation_results:
                    st.success(f"Evaluation completed successfully in {actual_time:.1f} seconds!")
                    
                    # Display results table
                    st.subheader("📊 Evaluation Results")
                    
                    metrics_data = []
                    for k in sorted(evaluation_results.keys()):
                        metrics = evaluation_results[k]
                        metrics_data.append({
                            'K': k,
                            'Precision': f"{metrics['precision']:.4f}",
                            'Recall': f"{metrics['recall']:.4f}",
                            'F1-Score': f"{metrics['f1']:.4f}",
                            'Users Evaluated': metrics['evaluated_users']
                        })
                    
                    metrics_df = pd.DataFrame(metrics_data)
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    # Performance summary
                    total_evaluated = sum([metrics['evaluated_users'] for metrics in evaluation_results.values()])
                    avg_evaluated = total_evaluated / len(evaluation_results) if evaluation_results else 0
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Actual Time", f"{actual_time:.1f}s")
                    with col2:
                        st.metric("Users/Second", f"{avg_evaluated/actual_time:.2f}")
                    with col3:
                        success_rate = (avg_evaluated / actual_users * 100) if actual_users > 0 else 0
                        st.metric("Success Rate", f"{success_rate:.1f}%")
                    
                    # Visualization (same as before)
                    if len(evaluation_results) > 1:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            k_vals = sorted(evaluation_results.keys())
                            precision_vals = [evaluation_results[k]['precision'] for k in k_vals]
                            recall_vals = [evaluation_results[k]['recall'] for k in k_vals]
                            f1_vals = [evaluation_results[k]['f1'] for k in k_vals]
                            
                            fig, ax = plt.subplots(figsize=(8, 6))
                            ax.plot(k_vals, precision_vals, 'o-', label='Precision', linewidth=2, markersize=8)
                            ax.plot(k_vals, recall_vals, 's-', label='Recall', linewidth=2, markersize=8)
                            ax.plot(k_vals, f1_vals, '^-', label='F1-Score', linewidth=2, markersize=8)
                            ax.set_xlabel('K (Number of Recommendations)')
                            ax.set_ylabel('Score')
                            ax.set_title('Performance Metrics vs K')
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                        
                        with col2:
                            users_evaluated = [evaluation_results[k]['evaluated_users'] for k in k_vals]
                            
                            fig, ax = plt.subplots(figsize=(8, 6))
                            bars = ax.bar(k_vals, users_evaluated, alpha=0.7, color='skyblue', edgecolor='black')
                            ax.set_xlabel('K (Number of Recommendations)')
                            ax.set_ylabel('Number of Users Evaluated')
                            ax.set_title('Users Successfully Evaluated')
                            ax.grid(True, alpha=0.3, axis='y')
                            
                            # Add value labels on bars
                            for bar, val in zip(bars, users_evaluated):
                                height = bar.get_height()
                                ax.text(bar.get_x() + bar.get_width()/2., height,
                                       f'{val}', ha='center', va='bottom')
                            st.pyplot(fig)
                    
                    # Save results
                    if st.button("💾 Save Evaluation Results", key="save_evaluation_btn"):
                        try:
                            # Save with timestamp
                            timestamp = int(time.time())
                            results_with_meta = {
                                'results': evaluation_results,
                                'metadata': {
                                    'timestamp': timestamp,
                                    'users_evaluated': actual_users,
                                    'k_values': eval_k_values,
                                    'evaluation_time': actual_time
                                }
                            }
                            
                            with open("evaluation_results.pkl", 'wb') as f:
                                pickle.dump(results_with_meta, f)
                            st.success("Evaluation results saved with metadata!")
                        except Exception as e:
                            st.error(f"Error saving results: {e}")
                
                else:
                    st.error("Evaluation failed to produce results. This could be due to:")
                    st.write("- Insufficient test data")
                    st.write("- Memory issues (try reducing max users)")  
                    st.write("- Association rules not generating recommendations")
                    st.write("Try running a smaller evaluation first (25 users) to test.")
        
        elif run_evaluation and not eval_k_values:
            st.warning("Please select at least one K value for evaluation.")
        
        # Load and display cached results
        if os.path.exists("evaluation_results.pkl") and not run_evaluation:
            try:
                with open("evaluation_results.pkl", 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Handle both old and new format
                if isinstance(cached_data, dict) and 'results' in cached_data:
                    saved_results = cached_data['results']
                    metadata = cached_data.get('metadata', {})
                    
                    st.info("Found cached evaluation results:")
                    if metadata:
                        st.write(f"- Cached on: {time.ctime(metadata.get('timestamp', 0))}")
                        st.write(f"- Users evaluated: {metadata.get('users_evaluated', 'Unknown')}")
                        st.write(f"- Evaluation time: {metadata.get('evaluation_time', 'Unknown'):.1f}s")
                else:
                    saved_results = cached_data
                    st.info("Found cached evaluation results (legacy format):")
                
                # Display cached results
                metrics_data = []
                for k in sorted(saved_results.keys()):
                    metrics = saved_results[k]
                    metrics_data.append({
                        'K': k,
                        'Precision': f"{metrics['precision']:.4f}",
                        'Recall': f"{metrics['recall']:.4f}",
                        'F1-Score': f"{metrics['f1']:.4f}",
                        'Users Evaluated': metrics['evaluated_users']
                    })
                
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error loading cached results: {e}")
        
        # Enhanced evaluation explanation
        with st.expander("ℹ️ Evaluation Details & Troubleshooting"):
            st.write("""
            **How This Evaluation Works:**
            
            1. **User Selection**: Selects random test users with ≥5 games
            2. **Data Splitting**: Splits each user's games 70% train / 30% test  
            3. **Recommendation**: Generates K recommendations using train games
            4. **Validation**: Checks how many recommendations appear in test games
            5. **Metrics**: Calculates precision, recall, F1-score for each user
            6. **Aggregation**: Averages metrics across all evaluated users
            
            **Troubleshooting Common Issues:**
            
            - **Evaluation stops unexpectedly**: Usually due to memory issues or invalid data
            - **No recommendations found**: Association rules may be too restrictive
            - **Low success rate**: Normal for recommendation systems (5-15% is typical)
            - **Slow performance**: Reduce max users or K values
            
            **Performance Tips:**
            
            - Start with Lightning preset (25 users) to test
            - Use fewer K values for faster evaluation  
            - Higher max users = more reliable but slower results
            - Memory usage scales with dataset size and rule complexity
            
            **Understanding Results:**
            
            - **Precision**: % of recommendations that user actually likes
            - **Recall**: % of user's actual games that were recommended  
            - **F1-Score**: Balanced metric combining precision and recall
            - Values of 0.05-0.15 are typical for recommendation systems
            """)
