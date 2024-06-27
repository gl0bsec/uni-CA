#%%
from mc_api import*
import mediacloud.api as mapi
# sources_ids = [3]
my_query = '"climate change"' # note the double quotes used to indicate use of the whole phrase
start_date = dt.date(2023, 11, 1)
end_date = dt.date(2023, 12,1)
df, source = get_timesries('"climate change"',start_date,end_date)
plot_series(df)

#%% 
fetch_all_stories_with_progress(my_query,start_date,end_date)

# %%
