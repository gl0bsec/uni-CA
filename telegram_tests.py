# Replace these with your own values
from telegram import*

# Keyword to search for
keyword = ''
# Minimum participants count
min_participants_count = 1000
# Number of top groups to return
top_n = 1000
# Limit of search results
search_limit = 1000

# Run the async function
channels = asyncio.run(main(keyword, min_participants_count, top_n, search_limit))

# Create a pandas DataFrame
df = pd.DataFrame(channels)
print(df)

# Save the DataFrame to a CSV file
df.to_csv('telegram_channels.csv', index=False)