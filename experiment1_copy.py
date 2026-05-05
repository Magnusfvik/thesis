# Quick check: How many users achieve < 1.0 at optimal α?
optimal_alpha = 0.35
users_below_perfect = []

for user_id in all_users:
    s = serendipity[optimal_alpha][user_id]
    if s < 1.0:
        users_below_perfect.append((user_id, s))

print(f"Users with serendipity < 1.0 at α=0.35: {len(users_below_perfect)}")
print(f"Details: {users_below_perfect}")