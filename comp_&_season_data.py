from statsbombpy import sb

# Fetch the master list of free competitions
competitions = sb.competitions()

# Print just the columns you need to see
print(competitions[['competition_id', 'competition_name', 'season_id', 'season_name']])