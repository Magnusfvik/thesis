# sensitivity_quick.py - Kjører på 10 sekunder!

import pandas as pd
import numpy as np
import pickle

# ============================================================================
# LOAD PRE-COMPUTED RECOMMENDATIONS
# ============================================================================

# Lagre recommendations fra PSOMOO.py først:
# pickle.dump(recommendations, open('recommendations.pkl', 'wb'))
# pickle.dump(user_E_u, open('user_E_u.pkl', 'wb'))

recommendations = pickle.load(open('recommendations.pkl', 'rb'))
user_E_u = pickle.load(open('user_E_u.pkl', 'rb'))

alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
alpha_names = {
    0.0: 'Pure CF (α=0.0)',
    0.25: 'CF-Biased (α=0.25)',
    0.5: 'Balanced (α=0.5)',
    0.75: 'Distance-Biased (α=0.75)',
    1.0: 'Pure Distance (α=1.0)'
}

# ============================================================================
# SENSITIVITY FUNCTIONS (kun serendipitet!)
# ============================================================================

def evaluate_serendipity_with_threshold(recommendations_dict, user_E_u, 
                                       distance_threshold, cf_threshold):
    """Serendipity with configurable thresholds"""
    serendipity_scores = []
    
    for user_id, recs in recommendations_dict.items():
        serendipitous = 0
        
        for rec in recs:
            is_unexpected = rec['distance'] > distance_threshold
            is_relevant = rec['cf_score'] > cf_threshold
            
            if is_unexpected and is_relevant:
                serendipitous += 1
        
        serendipity_scores.append(serendipitous / len(recs) if recs else 0)
    
    return np.mean(serendipity_scores)


def run_sensitivity_analysis(recommendations, user_E_u, alphas, alpha_names):
    """Test multiple threshold combinations"""
    threshold_combos = [
        (0.5, 2.5, "Lav distance, høy CF"),
        (0.6, 2.0, "Medium-lav"),
        (0.7, 1.8, "Original (Ge et al.)"),
        (0.8, 1.5, "Høy distance, lav CF"),
        (0.9, 1.2, "Ekstremt høy distance"),
        (0.65, 1.9, "Litt strengere enn Ge et al."),
        (0.70, 1.8, "Original (Ge et al.)"),  
        (0.75, 1.7, "Litt mildere enn Ge et al."),
        (0.70, 1.6, "Samme distance, lavere CF"),
        (0.70, 2.0, "Samme distance, høyere CF"),
    ]
    
    print("\n" + "=" * 80)
    print("SENSITIVITY ANALYSIS: Testing Different Serendipity Thresholds")
    print("=" * 80)
    
    sensitivity_results = []
    
    for dist_thresh, cf_thresh, label in threshold_combos:
        print(f"\n📊 Testing: distance > {dist_thresh}, CF > {cf_thresh} ({label})")
        
        combo_results = {
            'distance_threshold': dist_thresh,
            'cf_threshold': cf_thresh,
            'label': label,
            'serendipity_scores': {}
        }
        
        for alpha in alphas:
            serendipity = evaluate_serendipity_with_threshold(
                recommendations[alpha], 
                user_E_u, 
                dist_thresh, 
                cf_thresh
            )
            combo_results['serendipity_scores'][alpha] = serendipity
            print(f"  {alpha_names[alpha]:25s}: {serendipity:.3f}")
        
        winner_alpha = max(combo_results['serendipity_scores'].items(), 
                          key=lambda x: x[1])
        combo_results['winner'] = winner_alpha[0]
        combo_results['winner_score'] = winner_alpha[1]
        
        print(f"  🏆 Winner: {alpha_names[winner_alpha[0]]}")
        
        sensitivity_results.append(combo_results)
    
    return sensitivity_results


def create_sensitivity_table(sensitivity_results, alphas, alpha_names):
    """Create comparison table"""
    print("\n" + "=" * 80)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("=" * 80)
    
    rows = []
    for result in sensitivity_results:
        row = {
            'Terskler': f"({result['distance_threshold']}, {result['cf_threshold']})",
            'Beskrivelse': result['label']
        }
        
        for alpha in alphas:
            row[f'α={alpha}'] = result['serendipity_scores'][alpha]
        
        row['Vinner'] = f"α={result['winner']}"
        rows.append(row)
    
    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    
    # Count wins
    print("\n📈 ROBUSTNESS CHECK:")
    winner_counts = {}
    for result in sensitivity_results:
        winner = result['winner']
        winner_counts[winner] = winner_counts.get(winner, 0) + 1
    
    for alpha in sorted(winner_counts.keys()):
        count = winner_counts[alpha]
        total = len(sensitivity_results)
        print(f"  {alpha_names[alpha]:25s}: {count}/{total} ({100*count/total:.0f}%)")
    
    original_winner = sensitivity_results[2]['winner']
    most_frequent_winner = max(winner_counts.items(), key=lambda x: x[1])[0]
    
    if original_winner == most_frequent_winner:
        print(f"\n✅ ROBUST: {alpha_names[original_winner]} vinner konsistent!")
    else:
        print(f"\n⚠️  THRESHOLD-SENSITIVE: Vinneren varierer")
    
    return df


# ============================================================================
# RUN ANALYSIS
# ============================================================================

sensitivity_results = run_sensitivity_analysis(recommendations, user_E_u, 
                                               alphas, alpha_names)

sensitivity_df = create_sensitivity_table(sensitivity_results, alphas, alpha_names)

sensitivity_df.to_csv('sensitivity_analysis_results.csv', index=False)
print("\n✓ Saved: sensitivity_analysis_results.csv")