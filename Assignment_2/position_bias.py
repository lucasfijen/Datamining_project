#%% Imports
import pandas as pd 
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
sns.set_palette("Paired")
from matplotlib import rcParams
rcParams['font.family'] = 'serif'

# df = pd.read_csv('Assignment_2/data/training_set_VU_DM.csv')

def get_corrected_gain(df, correction_df):
    if correction_df is not None:
        print('apply existing correction')
    else:
        print('make new correction df')
        # Position bias relative chance of positions beeing clicked calculated
        random = df[df['random_bool'] == 1]
        non_random = df[df['random_bool'] == 0]

        random_clicks = random.groupby('position')['click_bool'].mean()
        random_bookings = random.groupby('position')['booking_bool'].mean()
        # random_ = pd.concat([random_clicks.rename('clicks'), random_bookings.rename('bookings')], axis=1)

        nonrandom_clicks = non_random.groupby('position')['click_bool'].mean()
        nonrandom_bookings = non_random.groupby('position')['booking_bool'].mean()
        # nonrandom_ = pd.concat([nonrandom_clicks.rename('clicks'), nonrandom_bookings.rename('bookings')], axis=1)

        # Calculate the relative chances of a click being performed due to click bias
        # clicks_correction = random_clicks / nonrandom_clicks
        # bookings_correction = random_bookings / nonrandom_bookings
        clicks_correction = 1 - random_clicks
        bookings_correction = 1 - random_bookings
        correction_df = pd.concat([random_clicks.rename('random clicks'), nonrandom_clicks.rename('non_random_clicks'), random_bookings.rename('random bookings'), nonrandom_bookings.rename('nonrandom bookings'), clicks_correction.rename('clicks_correction'), bookings_correction.rename('bookings_correction')], axis=1)
        correction_df.reset_index(inplace=True)
        correction_df.to_pickle('position_bias_correction.pickle')
    
    # print(correction_df.columns)
    
    # MAKE NEW DATAFRAME TO STORE CORRECTED GAIN
    corrected_gain = pd.DataFrame(df[['position']].copy())
    corrected_gain['total_non_corrected_gain'] = df['click_bool'] + df['booking_bool']
    corrected_gain['corrected_click_gain'] = df['click_bool'].copy()
    corrected_gain['corrected_book_gain'] = df['booking_bool'].copy() * 5

    # CORRECT THE GAIN
    for p in range(max(correction_df.position.values)):
        click_correction = correction_df[correction_df['position'] == p]['clicks_correction'].values.squeeze()
        book_correction = correction_df[correction_df['position'] == p]['bookings_correction'].values.squeeze()
        # print(p, click_correction, book_correction)
        corrected_gain.loc[corrected_gain['position'] == p, 'corrected_click_gain'] *= click_correction
        corrected_gain.loc[corrected_gain['position'] == p, 'corrected_book_gain'] *= book_correction

    corrected_gain['total_corrected_gain'] = corrected_gain['corrected_click_gain'] + corrected_gain['corrected_book_gain']
    corrected_gain = corrected_gain.drop(['position'], axis=1)
    # SAVE
    # corrected_gain.to_pickle('Assignment_2/data/corrected_gain_df.pkl')
    return correction_df, corrected_gain

#%% PLOT RANDOM VERSUS NON-RANDOM FRACTION OF CLICKS & BOOKINGS PER POSITION
# all = pd.concat([random_clicks.rename('random clicks'), nonrandom_clicks.rename('non random clicks'), random_bookings.rename('random bookings'), nonrandom_bookings.rename('non random_bookings')], axis=1)
# all_plot = all.plot.bar(figsize=[14, 10], width=1, color=sns.color_palette("Paired")[:4], linewidth=0)
# all_plot.set_ylabel('fraction', fontsize=14)
# all_plot.set_xlabel('position', fontsize=14)
# all_plot.set_title('Fraction of clicks and bookings at positions of the result page', fontsize=20)
# all_fig = all_plot.get_figure()
# all_fig.savefig('random_vs_nonrandom.png', dpi=200, facecolor='white')